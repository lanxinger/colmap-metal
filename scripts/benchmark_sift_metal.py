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
    parser.add_argument("image_path", type=Path, help="Directory of input images")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("/tmp/colmap-metal-sift-benchmark"),
        help="Scratch workspace to recreate for the benchmark",
    )
    parser.add_argument(
        "--colmap",
        type=Path,
        default=Path("build/src/colmap/exe/colmap"),
        help="COLMAP executable to benchmark",
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
            "keypoints": db_scalar(conn, "select coalesce(sum(rows), 0) from keypoints"),
            "match_pairs": db_scalar(conn, "select count(*) from matches"),
            "raw_matches": db_scalar(conn, "select coalesce(sum(rows), 0) from matches"),
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
    model_path = sparse_path / "0"
    analyzer_metrics = {}
    analyzer_seconds = None
    if model_path.exists():
        analyzer = run_command(
            f"model_analyzer {workspace.name}",
            [str(colmap), "model_analyzer", "--path", str(model_path)],
            cwd,
        )
        analyzer_seconds = analyzer["seconds"]
        analyzer_metrics = parse_model_analyzer(str(analyzer["output"]))

    return {
        "mapper_seconds": mapper["seconds"],
        "model_path": str(model_path),
        "model_analyzer_seconds": analyzer_seconds,
        "model_analyzer": analyzer_metrics,
    }


def main() -> int:
    args = parse_args()
    repo_root = Path.cwd()
    colmap = args.colmap
    if not colmap.is_absolute():
        colmap = repo_root / colmap
    image_path = args.image_path
    if not image_path.is_absolute():
        image_path = repo_root / image_path

    if not colmap.exists():
        raise FileNotFoundError(f"COLMAP executable not found: {colmap}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image path not found: {image_path}")

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
        ],
        repo_root,
    )

    result: dict[str, object] = {
        "image_path": str(image_path),
        "workspace": str(args.workspace),
        "guided_matching": args.guided,
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
        f"({float(before_seconds):.2f}s cpu / {float(after_seconds):.2f}s metal)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1)
