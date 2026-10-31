"""CPU geometry regressions; run with python -m unittest discover -s tests -v."""

import importlib.util
from pathlib import Path
import unittest

import numpy as np
from skimage import measure
import torch


# Import the extractor without core.__init__ loading unrelated model pipelines.
MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "baselines/core/models/autoencoders/surface_extractors.py"
)
spec = importlib.util.spec_from_file_location("surface_extractors", MODULE_PATH)
surface_extractors = importlib.util.module_from_spec(spec)
spec.loader.exec_module(surface_extractors)


def coordinates(bounds, shape):
    if isinstance(bounds, float):
        bounds = [-bounds] * 3 + [bounds] * 3
    lower, upper = np.asarray(bounds[:3]), np.asarray(bounds[3:])
    axes = [np.linspace(lo, hi, n, dtype=np.float32)
            for lo, hi, n in zip(lower, upper, shape)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1), lower, upper


class MarchingCubesCoordinatesTest(unittest.TestCase):
    def setUp(self):
        self.extractor = surface_extractors.MCSurfaceExtractor()

    def extract(self, field, bounds, resolution, level=0.0):
        return self.extractor.run(
            torch.from_numpy(np.ascontiguousarray(field, dtype=np.float32)),
            mc_level=level, bounds=bounds, octree_resolution=resolution,
        )

    def test_planes_preserve_position_and_boundary_extent(self):
        for bounds in (1.0, 1.01, [2.0, -3.0, 0.5, 6.0, 1.0, 2.5]):
            for resolution in (8, 31):
                xyz, lower, upper = coordinates(bounds, (resolution + 1,) * 3)
                for axis in range(3):
                    with self.subTest(bounds=bounds, resolution=resolution, axis=axis):
                        position = lower[axis] + 0.4 * (upper[axis] - lower[axis])
                        vertices, faces = self.extract(
                            xyz[..., axis] - position, bounds, resolution,
                        )
                        self.assertGreater(len(faces), 0)
                        np.testing.assert_allclose(vertices[:, axis], position, atol=2e-6, rtol=0)
                        for tangent_axis in set(range(3)) - {axis}:
                            np.testing.assert_allclose(
                                [vertices[:, tangent_axis].min(), vertices[:, tangent_axis].max()],
                                [lower[tangent_axis], upper[tangent_axis]], atol=2e-6, rtol=0,
                            )

    def test_rectangular_grids_use_each_axis_spacing(self):
        bounds = [-2.0, 1.0, -0.5, 4.0, 5.0, 1.5]
        xyz, lower, upper = coordinates(bounds, (9, 17, 33))
        for axis in range(3):
            with self.subTest(axis=axis):
                position = lower[axis] + 0.3 * (upper[axis] - lower[axis])
                vertices, _ = self.extract(xyz[..., axis] - position, bounds, 32)
                np.testing.assert_allclose(vertices[:, axis], position, atol=2e-6, rtol=0)

    def test_actual_grid_shape_takes_precedence_over_requested_resolution(self):
        # Adaptive decoders can produce a grid different from the requested size.
        xyz, _, _ = coordinates(1.0, (17, 17, 17))
        for requested in (16, 32, 64):
            with self.subTest(requested=requested):
                vertices, _ = self.extract(xyz[..., 0] - 0.25, 1.0, requested)
                np.testing.assert_allclose(vertices[:, 0], 0.25, atol=1e-6, rtol=0)

    def test_nonzero_isovalue_is_preserved(self):
        xyz, _, _ = coordinates(1.0, (17, 17, 17))
        vertices, _ = self.extract(xyz[..., 2], 1.0, 16, level=0.25)
        np.testing.assert_allclose(vertices[:, 2], 0.25, atol=1e-6, rtol=0)

    def test_spheres_preserve_center_and_radius(self):
        cases = (
            (1.0, 32, [0.25, -0.125, 0.0625], 0.5),
            (1.0, 64, [0.25, -0.125, 0.0625], 0.5),
            ([2.0, -3.0, 4.0, 6.0, 1.0, 8.0], 32, [4.0, -1.0, 6.0], 0.75),
        )
        for bounds, resolution, center, radius in cases:
            with self.subTest(bounds=bounds, resolution=resolution):
                xyz, lower, upper = coordinates(bounds, (resolution + 1,) * 3)
                field = np.linalg.norm(xyz - center, axis=-1) - radius
                vertices, _ = self.extract(field, bounds, resolution)
                # These centers and axial sphere extrema lie on grid samples.
                np.testing.assert_allclose(
                    (vertices.min(axis=0) + vertices.max(axis=0)) / 2,
                    center, atol=2e-6, rtol=0,
                )
                spacing = ((upper - lower) / resolution).max()
                radial_error = np.abs(np.linalg.norm(vertices - center, axis=-1) - radius)
                self.assertLess(radial_error.max(), spacing ** 2 / radius)

    def test_batch_wrapper_preserves_coordinates_and_output_types(self):
        xyz, _, _ = coordinates(1.0, (17, 17, 17))
        fields = np.stack([xyz[..., 0] - 0.25, xyz[..., 2] + 0.5])
        outputs = self.extractor(
            torch.from_numpy(fields), mc_level=0.0, bounds=1.0, octree_resolution=16,
        )
        self.assertEqual(len(outputs), 2)
        for output, axis, position in zip(outputs, (0, 2), (0.25, -0.5)):
            self.assertIsNotNone(output)
            self.assertEqual(output.mesh_v.dtype, np.float32)
            self.assertTrue(output.mesh_f.flags.c_contiguous)
            np.testing.assert_allclose(output.mesh_v[:, axis], position, atol=1e-6, rtol=0)

    def test_coordinate_mapping_preserves_marching_cubes_connectivity(self):
        xyz, _, _ = coordinates(1.0, (17, 17, 17))
        field = (np.linalg.norm(xyz, axis=-1) - 0.5).astype(np.float32)
        _, expected_faces, _, _ = measure.marching_cubes(field, 0.0, method="lewiner")
        _, actual_faces = self.extract(field, 1.0, 16)
        np.testing.assert_array_equal(actual_faces, expected_faces)


if __name__ == "__main__":
    unittest.main()
