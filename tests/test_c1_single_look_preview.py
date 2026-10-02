"""Synthetic interval-support proofs plus local private-profile preview checks."""
import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.c1 import c1_single_look_preview as p


class InterpolationTests(unittest.TestCase):
    def test_affine_lattice_exact_for_both_interpolations(self):
        coords = np.stack(np.meshgrid(*([np.linspace(0, 1, 3)]*3), indexing='ij'), axis=-1)
        matrix = np.array([[2., 3, 1], [1, -2, 4], [.2, 0, 1]])
        nodes = coords@matrix.T+[1, -3, .2]
        x = np.random.default_rng(1).random((300, 3))
        for method in ('tetrahedral', 'trilinear'):
            np.testing.assert_allclose(p.interpolate_nodes(nodes, x, method), x@matrix.T+[1, -3, .2], atol=1e-14)

    def test_proxy_preimage_affine_fixture(self):
        coords = np.stack(np.meshgrid(*([np.linspace(0, 1, 5)]*3), indexing='ij'), axis=-1)
        matrix = np.array([[60., 20, 15], [-40, 80, -20], [25, 10, -80]])
        nodes = coords@matrix.T+[0, -10, 30]
        positions = np.random.default_rng(12).random((80, 3))
        target = positions@matrix.T+[0, -10, 30]
        recovered, actual, error = p.proxy_preimages(nodes, target)
        np.testing.assert_allclose(recovered, positions, atol=1e-10)
        self.assertLess(error.max(), 1e-10)

    def test_proxy_preimage_reports_outside_failures(self):
        coords = np.stack(np.meshgrid(*([np.linspace(0, 1, 3)]*3), indexing='ij'), axis=-1)
        _, _, error = p.proxy_preimages(coords*10, np.array([[50., 50, 50]]))
        self.assertGreater(error[0], 50)

    def test_exact_corners(self):
        nodes = np.random.default_rng(2).random((3, 3, 3, 3))
        x = np.stack(np.meshgrid(*([np.linspace(0, 1, 3)]*3), indexing='ij'), axis=-1)
        np.testing.assert_array_equal(p.interpolate_nodes(nodes, x.reshape(-1, 3)), nodes.reshape(-1, 3))

    def test_coordinate_validation(self):
        nodes = np.zeros((2, 2, 2, 3))
        for x in ([[2., 0, 0]], [[float('nan'), 0, 0]]):
            with self.assertRaises(ValueError): p.interpolate_nodes(nodes, x)
        with self.assertRaises(ValueError): p.interpolate_nodes(nodes, [[0, 0, 0]], 'unknown')

    def test_quantization_no_silent_clip(self):
        for lab in ([200., 0, 0], [50, 180, 0], [float('nan'), 0, 0]):
            with self.assertRaises(ValueError): p.quantize_lab(np.array([lab]))

    def test_quantization_bound(self):
        x = np.random.default_rng(3).random((100, 3))*[100, 200, 200]-[0, 100, 100]
        self.assertLessEqual(np.linalg.norm(p.quantize_lab(x)-x, axis=-1).max(), np.linalg.norm(p.LAB_HALF_CODE)+1e-12)


class IntervalTests(unittest.TestCase):
    def test_interval_contains_random_convex_lab_hull(self):
        rng = np.random.default_rng(4)
        corners = rng.random((2, 2, 2, 3))*[90, 180, 180]-[0, 90, 90]
        low, high = p.cell_linear_rgb_bounds(corners)
        w = rng.random((10000, 8)); w /= w.sum(axis=1)[:, None]
        lab = w@corners.reshape(-1, 3)
        lin = p.c1.color.decode(p.c1.color.lab_rgb(lab))
        self.assertTrue(np.all(lin >= low.reshape(3)-1e-12))
        self.assertTrue(np.all(lin <= high.reshape(3)+1e-12))

    def test_constant_middle_gray_cell_certified(self):
        lab = p.c1.color.rgb_lab(np.array([[.5, .5, .5]]))[0]
        nodes = np.broadcast_to(lab, (2, 2, 2, 3))
        low, high = p.cell_linear_rgb_bounds(nodes)
        self.assertTrue(np.all(low > 0)); self.assertTrue(np.all(high < 1))
        np.testing.assert_allclose(low, high, atol=1e-12)

    def test_outside_gamut_constant_cell_not_certified(self):
        lab = p.c1.color.rgb_lab(np.array([[1.3, .5, .5]]))[0]
        nodes = np.broadcast_to(lab, (2, 2, 2, 3))
        low, high = p.cell_linear_rgb_bounds(nodes)
        self.assertGreater(high.max(), 1)

    def test_expansion_conservative(self):
        lab = p.c1.color.rgb_lab(np.array([[.5, .5, .5]]))[0]
        nodes = np.broadcast_to(lab, (2, 2, 2, 3))
        a, b = p.cell_linear_rgb_bounds(nodes)
        c, d = p.cell_linear_rgb_bounds(nodes, p.LAB_HALF_CODE)
        self.assertTrue(np.all(c <= a)); self.assertTrue(np.all(d >= b))

    def test_bad_cell_disables_all_its_incident_vertices(self):
        mask = np.ones((3, 3, 3), bool); mask[1, 1, 1] = False
        allowed = p.incident_safe_nodes(mask)
        self.assertEqual((~allowed).sum(), 8)
        self.assertFalse(allowed[1:3, 1:3, 1:3].any())

    def test_every_modified_node_only_touches_safe_cells(self):
        safe = np.random.default_rng(5).random((6, 6, 6)) > .2
        allowed = p.incident_safe_nodes(safe)
        for bits in p.BITS:
            cut = tuple(slice(b, b+6) for b in bits)
            self.assertFalse(allowed[cut][~safe].any())

    def test_support_feather_two_rings(self):
        allowed = np.ones((7, 7, 7), bool)
        w = p.support_feather(allowed, 2)
        self.assertEqual(w[0, 3, 3], .5)
        self.assertEqual(w[3, 3, 3], 1)
        allowed[3, 3, 3] = False
        w = p.support_feather(allowed, 2)
        self.assertEqual(w[3, 3, 3], 0)
        self.assertEqual(w[2, 3, 3], .5)

    def test_feather_validation(self):
        for rings in (0, 5, .5):
            with self.assertRaises(ValueError): p.support_feather(np.ones((3, 3, 3), bool), rings)


@unittest.skipUnless(p.c1.SOURCE.exists() and p.c1.VIVID.exists(), 'Private resources absent')
class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = p.c1.SOURCE.read_bytes(); cls.lut = p.c1.native_lut(cls.source)
        raw = p.c1.TextureLUT.from_cube(p.c1.VIVID)
        cls.table = p.c1.TextureLUT(raw.values.astype(np.float16).astype(float))
        cls.prepared = p.prepare(cls.lut, cls.table)
        cls.candidate = p.serialize(cls.source, cls.prepared['nodes'])
        cls.parsed = p.c1.audit.Lut16(dict(p.c1.read_tags(cls.candidate))[b'A2B0'])

    def test_pin_and_noncolor_structure_preserved(self):
        self.assertEqual(p.c1.sha(self.source), p.c1.SOURCE_SHA)
        self.assertEqual(self.source[4:128], self.candidate[4:128])
        old, new = dict(p.c1.read_tags(self.source)), dict(p.c1.read_tags(self.candidate))
        self.assertEqual(list(old), list(new))
        for k in (b'cprt', b'wtpt', b'tech'): self.assertEqual(old[k], new[k])
        self.assertNotEqual(old[b'A2B0'], new[b'A2B0'])
        self.assertNotEqual(old[b'desc'], new[b'desc'])

    def test_shapers_output_and_matrix_preserved(self):
        a, b = self.lut, self.parsed
        self.assertEqual(a.raw[a.ib:a.ie], b.raw[b.ib:b.ie])
        self.assertEqual(a.raw[a.ob:], b.raw[b.ob:])
        self.assertEqual(a.matrix, b.matrix)
        self.assertEqual(b.grid, 65)

    def test_safe_support_contains_real_vivid_effect(self):
        a = self.prepared
        self.assertGreater(a['safe_cells'].sum(), 0)
        self.assertGreater((a['total_weight'] == 1).sum(), 0)
        self.assertGreater(np.linalg.norm(a['nodes']-a['base'], axis=-1).max(), 1)

    def test_every_unsafe_cell_all_vertices_bit_identical(self):
        a = self.prepared
        same = np.all(a['nodes'] == a['base'], axis=-1)
        for bits in p.BITS:
            cut = tuple(slice(b, b+64) for b in bits)
            self.assertTrue(np.all(same[cut][~a['safe_cells']]))

    def test_unsafe_cell_offgrid_both_interpolations_exact(self):
        a = self.prepared
        coords = np.argwhere(~a['safe_cells'])
        rng = np.random.default_rng(6); coords = coords[rng.choice(len(coords), 300)]
        positions = (coords+rng.random((len(coords), 3)))/64
        for method in ('tetrahedral', 'trilinear'):
            x = p.interpolate_nodes(a['nodes'], positions, method)
            y = p.interpolate_nodes(a['base'], positions, method)
            np.testing.assert_array_equal(x, y)

    def test_native_vector_evaluator_matches_independent_scalar(self):
        q = np.random.default_rng(7).random((96, 3))
        nodes = p.c1.codes_lab(np.array(self.lut.clut).reshape(33, 33, 33, 3))
        for method in ('tetrahedral', 'trilinear'):
            actual = p.interpolate_nodes(nodes, p.shaped_coordinates(self.lut, q), method)
            expected = np.array([self.lut.evaluate(v, method)['pcs_Lab'] for v in q])
            np.testing.assert_allclose(actual, expected, atol=2e-13)

    def test_65_guard_includes_lab_half_code_expansion(self):
        base = self.prepared['base']
        low, high = p.cell_linear_rgb_bounds(base, p.LAB_HALF_CODE)
        guarded = np.all((low >= 1e-12) & (high <= 1-1e-12), axis=-1)
        np.testing.assert_array_equal(self.prepared['safe_cells'], guarded)
        lo0, hi0 = p.cell_linear_rgb_bounds(base)
        unguarded = np.all((lo0 >= 1e-12) & (hi0 <= 1-1e-12), axis=-1)
        self.assertTrue(np.all(~guarded | unguarded))
        self.assertGreater(int(unguarded.sum()), int(guarded.sum()))

    def test_65_does_not_claim_native_trilinear_equivalence(self):
        q = np.random.default_rng(719).random((4096, 3))
        shaped = p.shaped_coordinates(self.lut, q)
        old = p.c1.codes_lab(np.array(self.lut.clut).reshape(33, 33, 33, 3))
        native = p.interpolate_nodes(old, shaped, 'trilinear')
        refined = p.interpolate_nodes(self.prepared['base'], shaped, 'trilinear')
        self.assertGreater(np.linalg.norm(native-refined, axis=-1).max(), 1.)

    def test_native33_budget_preserves_base_and_bounded_leak_both_methods(self):
        a = p.prepare(self.lut, self.table, grid=33, rings=1, unsafe_budget=.75)
        native = p.c1.codes_lab(np.array(self.lut.clut).reshape(33, 33, 33, 3))
        np.testing.assert_array_equal(a['base'], native)
        q = np.random.default_rng(18).random((4096, 3))
        shaped = p.shaped_coordinates(self.lut, q)
        lo = np.minimum(np.floor(shaped*32).astype(int), 31)
        unsafe = ~a['safe_cells'][lo[:, 0], lo[:, 1], lo[:, 2]]
        for method in ('tetrahedral', 'trilinear'):
            b = p.interpolate_nodes(native, shaped, method)
            c = p.interpolate_nodes(a['nodes'], shaped, method)
            self.assertLessEqual(np.linalg.norm(c[unsafe]-b[unsafe], axis=-1).max(), .75+np.linalg.norm(p.LAB_HALF_CODE)+1e-10)
        serialized = p.serialize(self.source, a['nodes'])
        parsed = p.c1.audit.Lut16(dict(p.c1.read_tags(serialized))[b'A2B0'])
        self.assertEqual(parsed.raw[:parsed.cb], self.lut.raw[:self.lut.cb])
        self.assertEqual(parsed.grid, 33)

    def test_invalid_unsafe_budget_rejected(self):
        for value in (-1, 3, float('nan')):
            with self.assertRaises(ValueError):
                p.prepare(self.lut, self.table, grid=33, unsafe_budget=value)

    def test_original_is_unchanged(self):
        self.assertEqual(p.c1.SOURCE.read_bytes(), self.source)

    def test_actual_lcms_opens_and_evaluates(self):
        q = np.random.default_rng(8).random((48, 3))
        r = p.c1.audit.lcms_compare(self.candidate, q.tolist())
        if not r['available']: self.skipTest('System LCMS absent')
        self.assertTrue(r['intents']['perceptual']['transform_created'])
        actual = np.array(r['intents']['perceptual']['pcs_Lab'])
        expected = p.interpolate_nodes(self.prepared['nodes'], p.shaped_coordinates(self.lut, q))
        self.assertLess(np.linalg.norm(actual-expected, axis=-1).max(), .1)

    def test_existing_dir_refused_before_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError): p.build(Path(tmp), Path(tmp)/'release')
            self.assertFalse((Path(tmp)/'release').exists())


if __name__ == '__main__':
    unittest.main()
