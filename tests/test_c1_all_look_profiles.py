"""Synthetic gamut/PCS and native-structure tests for the complete C1 collection."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from experiments.c1 import all_look_profiles as p
from experiments.c1 import c1_single_profile as c1
from experiments.c1.look_composition import LookVariant, build_variant_plan


class GamutTests(unittest.TestCase):
    def test_standard_white_and_positive_luminance(self):
        for space in ('srgb', 'display-p3'):
            m = p.rgb_xyz_matrix(space)
            np.testing.assert_allclose(m @ np.ones(3), c1.color.WHITE65, atol=1e-14)
            self.assertTrue((m[1] > 0).all())
            self.assertAlmostEqual(m[1].sum(), 1)

    def test_in_gamut_is_bit_exact_identity(self):
        x = np.random.default_rng(1).random((500, 3))
        for space in ('srgb', 'display-p3'):
            y, changed = p.project_linear_rgb(x, p.rgb_xyz_matrix(space)[1])
            np.testing.assert_array_equal(x, y)
            self.assertFalse(changed.any())

    def test_projection_bounded_for_extended_values(self):
        x = np.random.default_rng(2).uniform(-3, 4, (10000, 3))
        for space in ('srgb', 'display-p3'):
            y, changed = p.project_linear_rgb(x, p.rgb_xyz_matrix(space)[1])
            self.assertTrue(((y >= 0) & (y <= 1)).all())
            self.assertEqual(int(changed.sum()), int(np.any((x < 0) | (x > 1), axis=1).sum()))

    def test_luminance_and_ray_direction_preserved_where_possible(self):
        weights = p.rgb_xyz_matrix('srgb')[1]
        x = np.array([[1.3, .2, -.1], [-.1, .6, .8], [.5, -.1, 2.]])
        neutral = x @ weights
        self.assertTrue(((neutral > 0) & (neutral < 1)).all())
        mapped, _ = p.project_linear_rgb(x, weights)
        np.testing.assert_allclose(mapped @ weights, neutral, atol=1e-14)
        a, b = x-neutral[:, None], mapped-neutral[:, None]
        np.testing.assert_allclose(np.cross(a, b), 0, atol=1e-14)

    def test_finite_and_weights_validation(self):
        for x in ([[float('nan'), 0, 0]], [[float('inf'), 0, 0]], [1, 2]):
            with self.assertRaises(ValueError):
                p.project_linear_rgb(x, [.2, .7, .1])
        for w in ([1, 1, 1], [-1, 1, 1], [1, 0]):
            with self.assertRaises(ValueError):
                p.project_linear_rgb([[0, 0, 0]], w)

    def test_gray_stays_gray(self):
        x = np.repeat(np.linspace(-.2, 1.2, 100)[:, None], 3, axis=1)
        for space in ('srgb', 'display-p3'):
            y, _ = p.project_linear_rgb(x, p.rgb_xyz_matrix(space)[1])
            np.testing.assert_allclose(y, np.clip(x, 0, 1), atol=1e-14)

    def test_p3_input_retains_colors_outside_srgb(self):
        rgb = np.array([[0., 1., 0.], [1., 0., 0.], [.2, .4, .8]])
        m = p.rgb_xyz_matrix('display-p3')
        lab = c1.color.xyz_lab(c1.color.decode(rgb) @ m.T @ np.linalg.inv(c1.color.CAT).T)
        source, _ = p.lab_to_source(lab, 'display-p3')
        recovered = p.color_spaces.convert_rgb(source, 'srgb', 'display-p3', clip=False)
        np.testing.assert_allclose(recovered, rgb, atol=1e-13)
        self.assertTrue(np.any((source < 0) | (source > 1)))

    def test_srgb_input_output_roundtrip(self):
        rgb = np.random.default_rng(3).uniform(.01, .99, (500, 3))
        lab = c1.color.rgb_lab(rgb)
        source, mask = p.lab_to_source(lab, 'srgb')
        recovered, _, _ = p.rendered_to_lab(source, 'srgb')
        self.assertFalse(mask.any())
        np.testing.assert_allclose(source, rgb, atol=2e-13)
        np.testing.assert_allclose(recovered, lab, atol=2e-12)


class LabRangeTests(unittest.TestCase):
    def test_in_range_unchanged(self):
        x = np.array([[0., 0., 0.], [100., 0., 0.], [30, -40, 20.]])
        y, reduced = p.fit_lab16(x)
        np.testing.assert_array_equal(x, y)
        self.assertFalse(reduced.any())

    def test_chroma_hue_and_luminance_retained(self):
        x = np.array([[50., 180., -90.], [30., -160., 180.], [99, 20., -145.]])
        y, reduced = p.fit_lab16(x)
        np.testing.assert_array_equal(y[:, 0], x[:, 0])
        np.testing.assert_allclose(y[:, 1]*x[:, 2]-y[:, 2]*x[:, 1], 0, atol=1e-11)
        self.assertTrue(reduced.all())
        codes = c1.lab_codes(y)
        self.assertTrue(((codes >= 0) & (codes <= 65535)).all())

    def test_invalid_lab_rejected(self):
        for x in ([[101, 0, 0]], [[-1, 0, 0]], [[10, float('nan'), 0]]):
            with self.assertRaises(ValueError):
                p.fit_lab16(x)

    def test_srgb_gray_rounds_to_neutral_lab(self):
        gray = np.repeat(np.linspace(0, 1, 257)[:, None], 3, axis=1)
        lab, _, _ = p.rendered_to_lab(gray, 'srgb')
        np.testing.assert_array_equal(np.floor(c1.lab_codes(lab)[:, 1:]+.5), 32768)


class CmmIntentTests(unittest.TestCase):
    def test_relative_intent_matches_raised_black_stored_pcs(self):
        from experiments.rendered_srgb.build_rendered_icc import profile_bytes
        coords = np.stack(np.meshgrid(*([np.linspace(0, 1, 2)]*3), indexing='ij'), axis=-1)
        def affine(rgb):
            return np.stack((5+90*np.mean(rgb, axis=-1),
                             30*(rgb[..., 0]-rgb[..., 1]),
                             20*(rgb[..., 1]-rgb[..., 2])), axis=-1)
        data = profile_bytes(affine(coords), 'Raised-black synthetic PCS fixture')
        points = np.array([[0, 0, 0], [1, 1, 1], [.1, .2, .3], [.5, .5, .5]])
        result = c1.audit.lcms_compare(data, points.tolist())
        if not result['available']:
            self.skipTest('LittleCMS2 runtime unavailable')
        expected = affine(points)
        relative = np.array(result['intents']['relative_colorimetric']['pcs_Lab'])
        self.assertLess(np.linalg.norm(relative-expected, axis=1).max(), .03)
        for entry in result['intents'].values():
            self.assertTrue(entry['transform_created'])
            self.assertTrue(np.isfinite(entry['pcs_Lab']).all())
        # Perceptual is recorded separately: its destination-normalized black
        # can differ from the literal stored PCS even when relative agrees.
        self.assertEqual(result['intents']['perceptual']['intent_number'], 0)


class CollectionTests(unittest.TestCase):
    def test_count_and_unique_names(self):
        plan = build_variant_plan()
        self.assertEqual(len(plan), 206)
        self.assertEqual(len({x.relative_path for x in plan}), 206)
        self.assertEqual(len({x.description for x in plan}), 206)

    def test_full_mono_does_not_restore_native_chroma(self):
        class GrayCompositor:
            bindings = {'gray': {'input_space': 'srgb', 'output_space': 'srgb'}}
            def apply(self, rgb, look, strength, color_filter):
                return np.repeat(np.mean(rgb, axis=1)[:, None], 3, axis=1)
        v = type('Variant', (), {'look_id': 'gray', 'strength': 100, 'color_filter': None})()
        x = np.array([[50., 100., -60.], [70., -80., 90.], [20., 20., 80.]])
        y, _ = p.render_pcs(x, GrayCompositor(), v)
        np.testing.assert_allclose(y[:, 1:], 0, atol=2e-12)
        self.assertGreater(np.linalg.norm(y-x, axis=1).max(), 10)

    def test_existing_output_refused_before_profile_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                p.build('missing-profile', 'missing-cubes', Path(tmp), Path(tmp)/'report.json')

    def test_existing_report_refused_before_profile_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp)/'existing.json'; report.write_text('{}')
            with self.assertRaises(FileExistsError):
                p.build('missing-profile', 'missing-cubes', Path(tmp)/'output', report)


if __name__ == '__main__':
    unittest.main()
