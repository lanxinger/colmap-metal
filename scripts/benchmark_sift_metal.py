#!/usr/bin/env python3
"""Benchmark CPU vs Metal SIFT matching on a COLMAP image dataset."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a repeatable COLMAP SIFT extraction/matching/mapper benchmark "
            "comparing CPU matching with Metal matching."
        )
    )
    parser.add_argument(
        "image_path", type=Path, help="Directory of input images"
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("/tmp/colmap-metal-sift-benchmark"),
        help="Scratch workspace to recreate for the benchmark",
    )
    parser.add_argument(
        "--colmap",
        type=Path,
        default=None,
        help="COLMAP executable to benchmark",
    )
    parser.add_argument(
        "--automatic",
        action="store_true",
        help="Benchmark automatic_reconstructor --dense 0 --use_gpu 1",
    )
    parser.add_argument(
        "--automatic-guided-matching",
        choices=("default", "on", "off"),
        default="default",
        help=(
            "Guided matching mode for --automatic. The default leaves COLMAP's "
            "automatic_reconstructor preset unchanged."
        ),
    )
    parser.add_argument(
        "--guided",
        action="store_true",
        help="Enable FeatureMatching.guided_matching for both matching runs",
    )
    parser.add_argument(
        "--skip-mapper",
        action="store_true",
        help="Skip mapper and model_analyzer steps",
    )
    parser.add_argument(
        "--skip-geometric-verification",
        action="store_true",
        help="Pass FeatureMatching.skip_geometric_verification to "
        "matching runs",
    )
    parser.add_argument(
        "--keep-workspace",
        action="store_true",
        help="Keep any existing workspace contents instead of recreating it",
    )
    parser.add_argument(
        "--single-camera",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pass ImageReader.single_camera to feature_extractor",
    )
    return parser.parse_args()


def run_command(label: str, command: list[str], cwd: Path) -> dict[str, object]:
    print(f"[benchmark] {label}", file=sys.stderr)
    start = time.perf_counter()
    result = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    duration = time.perf_counter() - start
    if result.returncode != 0:
        print(result.stdout, file=sys.stderr)
        raise RuntimeError(f"{label} failed with exit code {result.returncode}")
    return {
        "seconds": duration,
        "output": result.stdout,
    }


def default_colmap_path(repo_root: Path) -> Path:
    build_metal_colmap = repo_root / "build-metal/src/colmap/exe/colmap"
    if build_metal_colmap.exists():
        return build_metal_colmap
    return repo_root / "build/src/colmap/exe/colmap"


def backup_database(source: Path, destination: Path) -> None:
    if destination.exists():
        destination.unlink()
    with sqlite3.connect(source) as src, sqlite3.connect(destination) as dst:
        src.backup(dst)
    with sqlite3.connect(destination) as conn:
        conn.execute("delete from matches")
        conn.execute("delete from two_view_geometries")
        conn.commit()


def db_scalar(conn: sqlite3.Connection, sql: str) -> int:
    value = conn.execute(sql).fetchone()[0]
    return int(value or 0)


def db_stats(database: Path) -> dict[str, int]:
    with sqlite3.connect(database) as conn:
        return {
            "images": db_scalar(conn, "select count(*) from images"),
            "keypoints": db_scalar(
                conn, "select coalesce(sum(rows), 0) from keypoints"
            ),
            "match_pairs": db_scalar(conn, "select count(*) from matches"),
            "raw_matches": db_scalar(
                conn, "select coalesce(sum(rows), 0) from matches"
            ),
            "verified_pairs": db_scalar(
                conn, "select count(*) from two_view_geometries where rows > 0"
            ),
            "verified_inliers": db_scalar(
                conn, "select coalesce(sum(rows), 0) from two_view_geometries"
            ),
        }


def parse_model_analyzer(output: str) -> dict[str, str]:
    metrics = {}
    for line in output.splitlines():
        match = re.search(r"\] ([A-Za-z].*?): (.+)$", line)
        if not match:
            continue
        key = match.group(1).lower().replace(" ", "_")
        metrics[key] = match.group(2)
    return metrics


def parse_stage_minutes(output: str) -> dict[str, float]:
    stages = {}
    current_stage = None
    for line in output.splitlines():
        if "automatic_reconstruction.cc" in line and "=== " in line:
            stage_match = re.search(r"=== (.*?) ===", line)
            if stage_match:
                current_stage = stage_match.group(1).lower().replace(" ", "_")
            continue
        if current_stage and "timer.cc" in line and "Elapsed time:" in line:
            time_match = re.search(r"Elapsed time: ([0-9.]+) \[minutes\]", line)
            if time_match:
                stages[f"{current_stage}_minutes"] = float(time_match.group(1))
                current_stage = None
    return stages


def run_mapper_and_analyzer(
    colmap: Path, image_path: Path, workspace: Path, cwd: Path
) -> dict[str, object]:
    sparse_path = workspace / "sparse"
    sparse_path.mkdir(parents=True, exist_ok=True)
    mapper = run_command(
        f"mapper {workspace.name}",
        [
            str(colmap),
            "mapper",
            "--database_path",
            str(workspace / "database.db"),
            "--image_path",
            str(image_path),
            "--output_path",
            str(sparse_path),
        ],
        cwd,
    )
    result = {
        "mapper_seconds": mapper["seconds"],
    }
    result.update(run_largest_model_analyzer(colmap, sparse_path, cwd))
    return result


def run_largest_model_analyzer(
    colmap: Path, sparse_path: Path, cwd: Path
) -> dict[str, object]:
    if not sparse_path.exists():
        return {
            "model_path": str(sparse_path / "0"),
            "model_analyzer": {},
            "num_models": 0,
        }

    analyses = []
    model_paths = sorted(
        (
            path
            for path in sparse_path.iterdir()
            if path.is_dir() and path.name.isdigit()
        ),
        key=lambda path: int(path.name),
    )
    for model_path in model_paths:
        analyzer = run_command(
            f"model_analyzer {sparse_path.parent.name}/{model_path.name}",
            [str(colmap), "model_analyzer", "--path", str(model_path)],
            cwd,
        )
        metrics = parse_model_analyzer(str(analyzer["output"]))
        analyses.append((model_path, analyzer["seconds"], metrics))

    if not analyses:
        return {
            "model_path": str(sparse_path / "0"),
            "model_analyzer": {},
            "num_models": 0,
        }

    model_path, analyzer_seconds, analyzer_metrics = max(
        analyses,
        key=lambda analysis: (
            int(analysis[2].get("registered_images", 0)),
            int(analysis[2].get("points", 0)),
        ),
    )
    return {
        "model_path": str(model_path),
        "model_analyzer_seconds": analyzer_seconds,
        "model_analyzer": analyzer_metrics,
        "num_models": len(analyses),
    }


def run_automatic_benchmark(
    args: argparse.Namespace, colmap: Path, image_path: Path, repo_root: Path
) -> dict[str, object]:
    workspace = args.workspace
    if workspace.exists() and not args.keep_workspace:
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    command = [
        str(colmap),
        "automatic_reconstructor",
        "--workspace_path",
        str(workspace),
        "--image_path",
        str(image_path),
        "--dense",
        "0",
        "--use_gpu",
        "1",
    ]
    if args.automatic_guided_matching != "default":
        command.extend(
            [
                "--guided_matching",
                "1" if args.automatic_guided_matching == "on" else "0",
            ]
        )

    automatic = run_command(
        "automatic_reconstructor sparse metal", command, repo_root
    )
    result: dict[str, object] = {
        "mode": "automatic_reconstructor",
        "image_path": str(image_path),
        "workspace": str(workspace),
        "automatic_guided_matching": args.automatic_guided_matching,
        "total_seconds": automatic["seconds"],
        "stage_minutes": parse_stage_minutes(str(automatic["output"])),
        "db_stats": db_stats(workspace / "database.db"),
    }
    result.update(
        run_largest_model_analyzer(colmap, workspace / "sparse", repo_root)
    )
    return result


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd()
    colmap = args.colmap or default_colmap_path(repo_root)
    if not colmap.is_absolute():
        colmap = repo_root / colmap
    image_path = args.image_path
    if not image_path.is_absolute():
        image_path = repo_root / image_path

    if not colmap.exists():
        raise FileNotFoundError(f"COLMAP executable not found: {colmap}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image path not found: {image_path}")

    if args.automatic:
        result = run_automatic_benchmark(args, colmap, image_path, repo_root)
        print(json.dumps(result, indent=2, sort_keys=True))
        print(
            f"automatic sparse runtime: {float(result['total_seconds']):.2f}s",
            file=sys.stderr,
        )
        return 0

    if args.workspace.exists() and not args.keep_workspace:
        shutil.rmtree(args.workspace)
    before = args.workspace / "before_cpu"
    after = args.workspace / "after_metal"
    before.mkdir(parents=True, exist_ok=True)
    after.mkdir(parents=True, exist_ok=True)

    feature_command = [
        str(colmap),
        "feature_extractor",
        "--database_path",
        str(before / "database.db"),
        "--image_path",
        str(image_path),
        "--FeatureExtraction.use_gpu",
        "1",
    ]
    if args.single_camera:
        feature_command.extend(["--ImageReader.single_camera", "1"])
    extraction = run_command("feature_extractor", feature_command, repo_root)

    backup_database(before / "database.db", after / "database.db")

    guided_value = "1" if args.guided else "0"
    skip_geometric_verification_value = (
        "1" if args.skip_geometric_verification else "0"
    )
    before_matching = run_command(
        "exhaustive_matcher cpu",
        [
            str(colmap),
            "exhaustive_matcher",
            "--database_path",
            str(before / "database.db"),
            "--FeatureMatching.use_gpu",
            "0",
            "--FeatureMatching.guided_matching",
            guided_value,
            "--FeatureMatching.skip_geometric_verification",
            skip_geometric_verification_value,
        ],
        repo_root,
    )
    after_matching = run_command(
        "exhaustive_matcher metal",
        [
            str(colmap),
            "exhaustive_matcher",
            "--database_path",
            str(after / "database.db"),
            "--FeatureMatching.use_gpu",
            "1",
            "--FeatureMatching.guided_matching",
            guided_value,
            "--FeatureMatching.skip_geometric_verification",
            skip_geometric_verification_value,
        ],
        repo_root,
    )

    result: dict[str, object] = {
        "image_path": str(image_path),
        "workspace": str(args.workspace),
        "guided_matching": args.guided,
        "skip_geometric_verification": args.skip_geometric_verification,
        "extraction_seconds": extraction["seconds"],
        "before_cpu": {
            "matching_seconds": before_matching["seconds"],
            "db_stats": db_stats(before / "database.db"),
        },
        "after_metal": {
            "matching_seconds": after_matching["seconds"],
            "db_stats": db_stats(after / "database.db"),
        },
    }

    if not args.skip_mapper:
        result["before_cpu"].update(
            run_mapper_and_analyzer(colmap, image_path, before, repo_root)
        )
        result["after_metal"].update(
            run_mapper_and_analyzer(colmap, image_path, after, repo_root)
        )

    before_seconds = result["before_cpu"]["matching_seconds"]
    after_seconds = result["after_metal"]["matching_seconds"]
    speedup = float(before_seconds) / float(after_seconds)
    result["matching_speedup"] = speedup

    print(json.dumps(result, indent=2, sort_keys=True))
    print(
        f"matching speedup: {speedup:.2f}x "
        f"({float(before_seconds):.2f}s cpu / "
        f"{float(after_seconds):.2f}s metal)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
