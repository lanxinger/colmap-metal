import pytest

import pycolmap


def test_view_graph_calibration_options_init():
    options = pycolmap.ViewGraphCalibrationOptions()
    assert options is not None


def test_global_mapper_options_init():
    options = pycolmap.GlobalMapperOptions()
    assert options is not None


def test_global_pipeline_options_init():
    options = pycolmap.GlobalPipelineOptions()
    assert options is not None


def test_global_pipeline_options_min_num_matches():
    options = pycolmap.GlobalPipelineOptions()
    options.min_num_matches = 20
    assert options.min_num_matches == 20


def test_global_pipeline_options_multiple_models():
    options = pycolmap.GlobalPipelineOptions()
    assert options.multiple_models is False
    assert options.min_model_size == 3
    options.multiple_models = True
    options.min_model_size = 5
    assert options.multiple_models is True
    assert options.min_model_size == 5


@pytest.mark.parametrize(
    "name",
    [
        "incremental_mapping",
        "global_mapping",
        "triangulate_points",
        "calibrate_view_graph",
        "bundle_adjustment",
    ],
)
def test_public_api_callable(name):
    assert callable(getattr(pycolmap, name))
