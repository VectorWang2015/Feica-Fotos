"""Independent synthetic controls and local Q1 numerical regression for C1 probe."""
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
from PIL import ImageCms

from experiments.c1 import c1_single_profile as c1


def synthetic_tiff(path, pixels, compression=1):
    """Tiny independent uncompressed RGB16 fixture with standard sRGB ICC."""
    h, w, _ = pixels.shape
    icc = ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
    # Entries with payloads encoded later; all arrays explicitly little-endian.
    entries = [(256, 4, 1, struct.pack('<I', w)), (257, 4, 1, struct.pack('<I', h)),
               (258, 3, 3, struct.pack('<3H', 16, 16, 16)),
               (259, 3, 1, struct.pack('<H', compression)), (262, 3, 1, struct.pack('<H', 2)),
               (273, 4, 1, bytes(4)), (274, 3, 1, struct.pack('<H', 1)),
               (277, 3, 1, struct.pack('<H', 3)), (278, 4, 1, struct.pack('<I', h)),
               (279, 4, 1, struct.pack('<I', pixels.size*2)),
               (284, 3, 1, struct.pack('<H', 1)), (34675, 7, len(icc), icc)]
    data = bytearray(b'II'+struct.pack('<HI', 42, 8)+struct.pack('<H', len(entries))+bytes(12*len(entries))+bytes(4))
    strip_entry = None
    for i, (tag, typ, count, raw) in enumerate(entries):
        entry = 10+12*i
        struct.pack_into('<HHI', data, entry, tag, typ, count)
        if len(raw) <= 4:
            data[entry+8:entry+12] = raw.ljust(4, b'\0')
        else:
            struct.pack_into('<I', data, entry+8, len(data))
            data.extend(raw)
        if tag == 273:
            strip_entry = entry+8
    struct.pack_into('<I', data, strip_entry, len(data))
    data.extend(pixels.astype('<u2').tobytes())
    path.write_bytes(data)


class MathTests(unittest.TestCase):
    def test_v2_lab_code_roundtrip(self):
        codes = np.array([[0, 32768, 32768], [65280, 65280, 65280], [65535, 0, 65535]])
        np.testing.assert_allclose(c1.lab_codes(c1.codes_lab(codes)), codes, atol=1e-10)
        np.testing.assert_allclose(c1.codes_lab(codes)[1], [100, 127, 127])

    def test_probe_affine_commutes_with_convex_interpolation(self):
        rng = np.random.default_rng(1)
        x = rng.random((8, 3))*[100, 200, 200]-[0, 100, 100]
        w = rng.random(8); w /= w.sum()
        np.testing.assert_allclose(c1.probe(w@x), w@c1.probe(x), atol=1e-12)

    def test_probe_invertible(self):
        x = np.array([[5, 12, -22], [50, -10, 30], [99, .2, .4]])
        np.testing.assert_allclose(c1.probe(x)@np.linalg.inv(c1.PROBE_MATRIX).T, x, atol=1e-12)

    def test_downstream_nonlinearity_does_not_commute(self):
        x = np.array([[50., 0, 0]])
        def d(a):
            out = a.copy(); out[:, 0] = 100*(out[:, 0]/100)**.8; return out
        self.assertGreater(np.linalg.norm(d(c1.probe(x))-c1.probe(d(x))), .1)

    def test_signed_colorimetric_roundtrip(self):
        x = np.array([[-.3, .8, 1.4], [0, 0, 0], [.5, .4, .3], [1, 1, 1]])
        np.testing.assert_allclose(c1.color.lab_rgb(c1.color.rgb_lab(x)), x, atol=2e-6)

    def test_compression_is_monotonic_and_reversible(self):
        x = np.linspace(-1, 2, 1001)
        y = c1.compress(x)
        self.assertTrue(np.all(np.diff(y) > 0))
        np.testing.assert_allclose(c1.decompress(y), x, atol=1e-6)

    def test_compression_preserves_core(self):
        x = np.linspace(.08, .92, 100)
        np.testing.assert_array_equal(c1.compress(x), x)

    def test_inverse_rejects_closed_cube_endpoints(self):
        for x in (0., 1., -1., float('nan')):
            with self.assertRaises(ValueError):
                c1.decompress(np.array([[x, .5, .5]]))

    def test_bounded_transform_exact_identity_outside(self):
        table = c1.TextureLUT(np.full((2, 2, 2, 3), .4))
        lab = c1.color.rgb_lab(np.array([[-.1, .5, .5], [.5, 1.2, .5]]))
        out, w = c1.bounded_look(lab, table)
        np.testing.assert_array_equal(out, lab)
        np.testing.assert_array_equal(w, [0, 0])

    def test_bounded_core_applies_table(self):
        table = c1.TextureLUT(np.full((2, 2, 2, 3), .4))
        lab = c1.color.rgb_lab(np.array([[.5, .5, .5]]))
        out, w = c1.bounded_look(lab, table)
        np.testing.assert_allclose(out, c1.color.rgb_lab(np.array([[.4, .4, .4]])), atol=1e-10)
        np.testing.assert_array_equal(w, [1])

    def test_bounded_feather_is_explicit(self):
        table = c1.TextureLUT(np.full((2, 2, 2, 3), .4))
        lab = c1.color.rgb_lab(np.array([[.04, .5, .5]]))
        out, w = c1.bounded_look(lab, table)
        self.assertAlmostEqual(w[0], .5, places=8)
        np.testing.assert_allclose(out, lab+.5*(c1.color.rgb_lab(np.array([[.4, .4, .4]]))-lab), atol=1e-8)

    def test_invalid_margin_rejected(self):
        with self.assertRaises(ValueError):
            c1.bounded_look(np.array([[50, 0, 0]]), None, margin=0)


@unittest.skipUnless(c1.SOURCE.exists(), 'Private user profile absent; synthetic tests remain portable')
class NativeProfileTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = c1.SOURCE.read_bytes()
        if c1.sha(cls.data) != c1.SOURCE_SHA:
            raise AssertionError('Unexpected user profile hash')
        cls.new = c1.serialize_probe(cls.data)
        cls.old_lut = c1.native_lut(cls.data)
        cls.new_lut = c1.native_lut(cls.new)

    def test_header_and_unrelated_tags_preserved(self):
        self.assertEqual(self.data[4:128], self.new[4:128])
        old, new = dict(c1.read_tags(self.data)), dict(c1.read_tags(self.new))
        self.assertEqual(list(old), list(new))
        for tag in (b'cprt', b'wtpt', b'tech'):
            self.assertEqual(old[tag], new[tag])
        self.assertNotEqual(old[b'desc'], new[b'desc'])
        self.assertNotEqual(old[b'A2B0'], new[b'A2B0'])

    def test_shapers_matrix_output_tables_preserved(self):
        a, b = self.old_lut, self.new_lut
        self.assertEqual(a.raw[:a.cb], b.raw[:b.cb])
        self.assertEqual(a.raw[a.ce:], b.raw[b.ce:])

    def test_affine_quantization_bound_off_nodes(self):
        points = np.random.default_rng(2).random((128, 3))
        for method in ('tetrahedral', 'trilinear'):
            expected = c1.probe(np.array([self.old_lut.evaluate(p.tolist(), method)['pcs_Lab'] for p in points]))
            actual = np.array([self.new_lut.evaluate(p.tolist(), method)['pcs_Lab'] for p in points])
            self.assertLessEqual(np.linalg.norm(actual-expected, axis=1).max(), np.linalg.norm([.5/652.8, .5/256, .5/256])+1e-10)

    def test_identity_replacement_preserves_a2b_bytes(self):
        nodes = c1.codes_lab(np.asarray(self.old_lut.clut).reshape(-1, 3))
        self.assertEqual(c1.replace_clut(self.old_lut, nodes), self.old_lut.raw)

    def test_native_grid_refinement_identity_bytes(self):
        self.assertEqual(c1.refined_payload(self.old_lut, 33, lambda lab: lab), self.old_lut.raw)

    def test_refinement_rejects_unbounded_grid(self):
        with self.assertRaises(ValueError):
            c1.refined_payload(self.old_lut, 257, lambda lab: lab)

    def test_unrepresentable_values_rejected(self):
        nodes = np.zeros((33**3, 3)); nodes[:, 0] = 200
        with self.assertRaises(ValueError):
            c1.replace_clut(self.old_lut, nodes)

    def test_independent_lcms_opens_and_evaluates(self):
        points = [[.1, .2, .3], [.5, .5, .5]]
        r = c1.audit.lcms_compare(self.new, points)
        if not r['available']:
            self.skipTest('LCMS unavailable')
        self.assertTrue(r['intents']['perceptual']['transform_created'])
        actual = np.array(r['intents']['perceptual']['pcs_Lab'])
        expected = c1.evaluate_lut(self.new_lut, points)
        self.assertLess(np.linalg.norm(actual-expected, axis=1).max(), .02)


class IOTests(unittest.TestCase):
    def test_json_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'out.json'
            c1.create_json(p, {'a': 1})
            with self.assertRaises(FileExistsError):
                c1.create_json(p, {'a': 2})
            self.assertEqual(json.loads(p.read_text()), {'a': 1})

    def test_rgb16_reader_preserves_codes_and_stride(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'fixture.tif'
            x = np.arange(5*7*3, dtype=np.uint16).reshape(5, 7, 3)*521
            synthetic_tiff(p, x)
            full, _ = c1.read_tiff(p, 1)
            sub, _ = c1.read_tiff(p, 2)
            np.testing.assert_array_equal(full, x)
            np.testing.assert_array_equal(sub, x[::2, ::2])

    def test_compressed_tiff_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/'fixture.tif'
            synthetic_tiff(p, np.ones((5, 7, 3), dtype=np.uint16), compression=5)
            with self.assertRaises(ValueError):
                c1.read_tiff(p)

    def test_compare_records_no_automatic_host_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            p, report = Path(tmp)/'fixture.tif', Path(tmp)/'report.json'
            synthetic_tiff(p, np.full((5, 7, 3), 32768, dtype=np.uint16))
            c1.compare(p, p, report, 1, p)
            r = json.loads(report.read_text())
            self.assertTrue(r['no_automatic_pass_fail'])
            self.assertEqual(r['baseline_repeat_deltaE76']['max'], 0)
            self.assertGreater(r['candidate_vs_post_export_H_deltaE76_nonclipped']['mean'], 1)


if __name__ == '__main__':
    unittest.main()
