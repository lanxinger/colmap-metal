import copy
import pickle

import numpy as np

import pycolmap


def test_two_view_geometry_configuration_undefined():
    assert pycolmap.TwoViewGeometryConfiguration.UNDEFINED is not None


def test_two_view_geometry_configuration_degenerate():
    assert pycolmap.TwoViewGeometryConfiguration.DEGENERATE is not None


def test_two_view_geometry_configuration_calibrated():
    assert pycolmap.TwoViewGeometryConfiguration.CALIBRATED is not None


def test_two_view_geometry_configuration_uncalibrated():
    assert pycolmap.TwoViewGeometryConfiguration.UNCALIBRATED is not None


def test_two_view_geometry_configuration_planar():
    assert pycolmap.TwoViewGeometryConfiguration.PLANAR is not None


def test_two_view_geometry_configuration_panoramic():
    assert pycolmap.TwoViewGeometryConfiguration.PANORAMIC is not None


def test_two_view_geometry_configuration_planar_or_panoramic():
    assert pycolmap.TwoViewGeometryConfiguration.PLANAR_OR_PANORAMIC is not None


def test_two_view_geometry_configuration_watermark():
    assert pycolmap.TwoViewGeometryConfiguration.WATERMARK is not None


def test_two_view_geometry_configuration_multiple():
    assert pycolmap.TwoViewGeometryConfiguration.MULTIPLE is not None


def test_homography_estimation_space():
    assert pycolmap.HomographyEstimationSpace.UNKNOWN is not None
    assert pycolmap.HomographyEstimationSpace.PIXEL is not None
    assert pycolmap.HomographyEstimationSpace.CAMERA_RAY is not None


def test_two_view_geometry_default_init():
    geometry = pycolmap.TwoViewGeometry()
    assert geometry is not None


def test_two_view_geometry_config_readwrite():
    geometry = pycolmap.TwoViewGeometry()
    geometry.config = pycolmap.TwoViewGeometryConfiguration.CALIBRATED
    assert geometry.config == pycolmap.TwoViewGeometryConfiguration.CALIBRATED


def test_two_view_geometry_e_readwrite():
    geometry = pycolmap.TwoViewGeometry()
    essential = np.eye(3)
    geometry.E = essential
    np.testing.assert_array_almost_equal(geometry.E, essential)


def test_two_view_geometry_f_readwrite():
    geometry = pycolmap.TwoViewGeometry()
    fundamental = np.eye(3)
    geometry.F = fundamental
    np.testing.assert_array_almost_equal(geometry.F, fundamental)


def test_two_view_geometry_h_readwrite():
    geometry = pycolmap.TwoViewGeometry()
    geometry.H_estimation_space = pycolmap.HomographyEstimationSpace.CAMERA_RAY
    homography = np.eye(3)
    geometry.H = homography
    np.testing.assert_array_almost_equal(geometry.H, homography)
    assert (
        geometry.H_estimation_space
        == pycolmap.HomographyEstimationSpace.CAMERA_RAY
    )

    geometry.H[0, 0] = 2.0
    assert geometry.H[0, 0] == 2.0
    assert (
        geometry.H_estimation_space
        == pycolmap.HomographyEstimationSpace.CAMERA_RAY
    )

    geometry.H = 2.0 * homography
    np.testing.assert_array_almost_equal(geometry.H, 2.0 * homography)
    assert (
        geometry.H_estimation_space
        == pycolmap.HomographyEstimationSpace.CAMERA_RAY
    )


def test_two_view_geometry_h_provenance_init_order():
    geometry_h_first = pycolmap.TwoViewGeometry(
        H=np.eye(3),
        H_estimation_space=pycolmap.HomographyEstimationSpace.CAMERA_RAY,
    )
    geometry_space_first = pycolmap.TwoViewGeometry(
        {
            "H_estimation_space": pycolmap.HomographyEstimationSpace.CAMERA_RAY,
            "H": np.eye(3),
        }
    )

    assert (
        geometry_h_first.H_estimation_space
        == pycolmap.HomographyEstimationSpace.CAMERA_RAY
    )
    assert (
        geometry_space_first.H_estimation_space
        == pycolmap.HomographyEstimationSpace.CAMERA_RAY
    )


def test_two_view_geometry_h_provenance_mergedict_order():
    geometry_h_first = pycolmap.TwoViewGeometry(
        H=np.eye(3),
        H_estimation_space=pycolmap.HomographyEstimationSpace.PIXEL,
    )
    geometry_space_first = copy.deepcopy(geometry_h_first)

    geometry_h_first.mergedict(
        {
            "H": 2.0 * np.eye(3),
            "H_estimation_space": pycolmap.HomographyEstimationSpace.CAMERA_RAY,
        }
    )
    geometry_space_first.mergedict(
        {
            "H_estimation_space": pycolmap.HomographyEstimationSpace.CAMERA_RAY,
            "H": 2.0 * np.eye(3),
        }
    )

    for geometry in (geometry_h_first, geometry_space_first):
        np.testing.assert_array_almost_equal(geometry.H, 2.0 * np.eye(3))
        assert (
            geometry.H_estimation_space
            == pycolmap.HomographyEstimationSpace.CAMERA_RAY
        )


def test_two_view_geometry_h_provenance_dataclass_roundtrip():
    geometry = pycolmap.TwoViewGeometry()
    geometry.H = np.eye(3)
    geometry.H_estimation_space = pycolmap.HomographyEstimationSpace.CAMERA_RAY

    assert (
        geometry.todict()["H_estimation_space"]
        == pycolmap.HomographyEstimationSpace.CAMERA_RAY
    )
    assert (
        copy.deepcopy(geometry).H_estimation_space
        == pycolmap.HomographyEstimationSpace.CAMERA_RAY
    )
    assert (
        pickle.loads(pickle.dumps(geometry)).H_estimation_space
        == pycolmap.HomographyEstimationSpace.CAMERA_RAY
    )


def test_two_view_geometry_cam2_from_cam1_readwrite():
    geometry = pycolmap.TwoViewGeometry()
    rigid = pycolmap.Rigid3d()
    geometry.cam2_from_cam1 = rigid
    assert isinstance(geometry.cam2_from_cam1, pycolmap.Rigid3d)


def test_two_view_geometry_inlier_matches_readwrite():
    geometry = pycolmap.TwoViewGeometry()
    matches = np.array([[0, 1], [2, 3]], dtype=np.uint32)
    geometry.inlier_matches = matches
    result = geometry.inlier_matches
    assert result.shape[0] == 2


def test_two_view_geometry_tri_angle_readwrite():
    geometry = pycolmap.TwoViewGeometry()
    geometry.tri_angle = 1.5
    assert geometry.tri_angle == 1.5


def test_two_view_geometry_invert():
    geometry = pycolmap.TwoViewGeometry()
    geometry.config = pycolmap.TwoViewGeometryConfiguration.CALIBRATED
    geometry.invert()
    assert geometry is not None
