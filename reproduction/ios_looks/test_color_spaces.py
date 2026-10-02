#!/usr/bin/env python3
"""Independent synthetic standard-color tests; no App resources or images.

The W3C rational fixtures below are fixed published coefficients, NOT values
computed with the implementation's chromaticity builder or its inverses.
Source: https://www.w3.org/TR/css-color-4/#color-conversion-code
Functions lin_sRGB_to_XYZ, XYZ_to_lin_sRGB, lin_P3_to_XYZ, XYZ_to_lin_P3;
retrieved 2026-10-02. Transfer references are scalar, separately evaluated.
Run: /usr/bin/python3 reproduction/ios_looks/test_color_spaces.py
"""
import itertools
import math
import unittest

import numpy as np

if __package__:
    from .color_spaces import (
        COLOR_SPACES, convert_rgb, linear_rgb_matrix, linear_rgb_to_xyz_matrix,
        out_of_gamut, srgb_eotf, srgb_oetf,
    )
else:
    from color_spaces import (
        COLOR_SPACES, convert_rgb, linear_rgb_matrix, linear_rgb_to_xyz_matrix,
        out_of_gamut, srgb_eotf, srgb_oetf,
    )


# Fixed W3C rational coefficients: both directions supplied independently.
W3C_SRGB_TO_XYZ = np.array([
    [506752 / 1228815, 87881 / 245763, 12673 / 70218],
    [87098 / 409605, 175762 / 245763, 12673 / 175545],
    [7918 / 409605, 87881 / 737289, 1001167 / 1053270],
])
W3C_XYZ_TO_SRGB = np.array([
    [12831 / 3959, -329 / 214, -1974 / 3959],
    [-851781 / 878810, 1648619 / 878810, 36519 / 878810],
    [705 / 12673, -2585 / 12673, 705 / 667],
])
W3C_P3_TO_XYZ = np.array([
    [608311 / 1250200, 189793 / 714400, 198249 / 1000160],
    [35783 / 156275, 247089 / 357200, 198249 / 2500400],
    [0, 32229 / 714400, 5220557 / 5000800],
])
W3C_XYZ_TO_P3 = np.array([
    [446124 / 178915, -333277 / 357830, -72051 / 178915],
    [-14852 / 17905, 63121 / 35810, 423 / 17905],
    [11844 / 330415, -50337 / 660830, 316169 / 330415],
])
FIXTURE_MATRICES = {
    "srgb": (W3C_SRGB_TO_XYZ, W3C_XYZ_TO_SRGB),
    "display-p3": (W3C_P3_TO_XYZ, W3C_XYZ_TO_P3),
}


def scalar_decode(c):
    if abs(c) <= 0.04045:
        return c / 12.92
    return math.copysign(math.pow((abs(c) + 0.055) / 1.055, 2.4), c)


def scalar_encode(c):
    if abs(c) <= 0.0031308:
        return 12.92 * c
    return math.copysign(1.055 * math.pow(abs(c), 1.0 / 2.4) - 0.055, c)


def fixture_convert(rgb, source, target):
    """Scalar transfer + fixed forward AND reverse XYZ fixtures, no inverse."""
    src = source.removesuffix("-linear")
    dst = target.removesuffix("-linear")
    values = list(rgb)
    if not source.endswith("-linear"):
        values = [scalar_decode(float(c)) for c in values]
    xyz = [sum(row[j] * values[j] for j in range(3))
           for row in FIXTURE_MATRICES[src][0]]
    values = [sum(row[j] * xyz[j] for j in range(3))
              for row in FIXTURE_MATRICES[dst][1]]
    if not target.endswith("-linear"):
        values = [scalar_encode(float(c)) for c in values]
    return values


class ColorSpaceTests(unittest.TestCase):
    def test_chromaticity_derived_matrices_match_fixed_w3c_fixtures(self):
        for mode, expected in (("srgb-linear", W3C_SRGB_TO_XYZ),
                               ("display-p3-linear", W3C_P3_TO_XYZ)):
            with self.subTest(mode=mode):
                np.testing.assert_allclose(linear_rgb_to_xyz_matrix(mode), expected,
                                           rtol=0, atol=4e-16)

    def test_cross_primaries_matrices_match_fixed_w3c_coefficients(self):
        cases = [("srgb-linear", "display-p3-linear", W3C_XYZ_TO_P3 @ W3C_SRGB_TO_XYZ),
                 ("display-p3-linear", "srgb-linear", W3C_XYZ_TO_SRGB @ W3C_P3_TO_XYZ)]
        for source, target, expected in cases:
            with self.subTest(source=source):
                np.testing.assert_allclose(linear_rgb_matrix(source, target), expected,
                                           rtol=0, atol=9e-16)

    def test_d65_white_point_not_dci_white(self):
        white_xyz = [0.9504559270516716, 1.0, 1.0890577507598784]
        for mode in ("srgb-linear", "display-p3-linear"):
            np.testing.assert_allclose(linear_rgb_to_xyz_matrix(mode) @ np.ones(3),
                                       white_xyz, rtol=0, atol=5e-16)

    def test_linear_red_primary_xyz_known_vectors(self):
        np.testing.assert_allclose(linear_rgb_to_xyz_matrix("srgb-linear") @ [1, 0, 0],
                                   [0.4123907992659595, 0.21263900587151036,
                                    0.01933081871559185], rtol=0, atol=3e-16)
        np.testing.assert_allclose(linear_rgb_to_xyz_matrix("display-p3-linear") @ [1, 0, 0],
                                   [0.4865709486482162, 0.2289745640697488, 0.0],
                                   rtol=0, atol=3e-16)

    def test_all_sixteen_mode_pairs_against_external_coefficient_oracle(self):
        vectors = [[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                   [.25, .5, .75], [.04045, .18, .0031308], [-.3, .9, 1.4]]
        for source, target in itertools.product(COLOR_SPACES, repeat=2):
            with self.subTest(source=source, target=target):
                expected = [fixture_convert(v, source, target) for v in vectors]
                # Official rounded transfer joins are not exact inverses. Identity
                # conversion correctly skips the otherwise redundant round trip.
                tolerance = 3e-8 if source == target else 2e-14
                np.testing.assert_allclose(convert_rgb(vectors, source, target), expected,
                                           rtol=0, atol=tolerance)

    def test_standard_transfer_known_vectors(self):
        encoded = [[0, .04045, .5], [.003, 1, .25]]
        linear = [[0, .0031308049535603713, .21404114048223255],
                  [.0002321981424148607, 1, .05087608817155679]]
        np.testing.assert_allclose(srgb_eotf(encoded), linear, rtol=0, atol=2e-16)
        np.testing.assert_allclose(srgb_oetf([[.0031308, .18, .5]]),
                                   [[.040449936, .46135612950044164, .7353569830524495]],
                                   rtol=0, atol=2e-16)

    def test_display_p3_uses_srgb_transfer_not_gamma_26(self):
        rgb = [.04, .5, .75]
        expected = [.0030959752321981426, .21404114048223255, .5225215539683921]
        actual = convert_rgb(rgb, "display-p3", "display-p3-linear")
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-16)
        np.testing.assert_array_equal(actual, convert_rgb(rgb, "srgb", "srgb-linear"))
        self.assertGreater(abs(actual[1] - .5 ** 2.6), .04)

    def test_transfer_breakpoint_branches_and_negative_reflection(self):
        for transfer, threshold, reference in (
            (srgb_eotf, .04045, scalar_decode), (srgb_oetf, .0031308, scalar_encode)
        ):
            values = np.array([np.nextafter(threshold, 0), threshold,
                               np.nextafter(threshold, np.inf)])
            for sign in (1, -1):
                rgb = values * sign
                np.testing.assert_allclose(transfer(rgb), [reference(c) for c in rgb],
                                           rtol=0, atol=1e-17)

    def test_rounded_breakpoints_have_documented_roundtrip_tolerance(self):
        rgb = np.array([.04045, -.04045, 0])
        result = srgb_oetf(srgb_eotf(rgb))
        self.assertGreater(abs(result[0] - rgb[0]), 1e-8)
        np.testing.assert_allclose(result, rgb, rtol=0, atol=3e-8)

    def test_neutral_axis_preserved_across_primaries(self):
        rgb = np.repeat(np.array([-2, -.5, 0, .001, .18, .5, 1, 2])[:, None], 3, axis=1)
        for source, target in (("srgb", "display-p3"), ("display-p3", "srgb"),
                               ("srgb-linear", "display-p3-linear"),
                               ("display-p3-linear", "srgb-linear")):
            np.testing.assert_allclose(convert_rgb(rgb, source, target), rgb,
                                       rtol=0, atol=2e-15)

    def test_roundtrip_all_mode_pairs_in_gamut_and_extended(self):
        rng = np.random.default_rng(6182)
        samples = np.concatenate([rng.random((200, 3)), rng.uniform(-3, 4, (200, 3))])
        for source, target in itertools.product(COLOR_SPACES, repeat=2):
            with self.subTest(source=source, target=target):
                converted = convert_rgb(samples, source, target)
                restored = convert_rgb(converted, target, source)
                np.testing.assert_allclose(restored, samples, rtol=0, atol=4e-13)

    def test_p3_red_is_outside_srgb_without_default_clipping(self):
        linear = convert_rgb([1, 0, 0], "display-p3", "srgb-linear")
        np.testing.assert_allclose(linear,
                                   [1.2249401762805598, -.0420569547096882, -.01963755459033443],
                                   rtol=0, atol=9e-16)
        encoded = convert_rgb([1, 0, 0], "display-p3", "srgb")
        np.testing.assert_allclose(encoded,
                                   [1.0930663624351615, -.2267419735697543, -.1501345809371195],
                                   rtol=0, atol=3e-15)
        self.assertTrue(out_of_gamut(linear))
        self.assertTrue(out_of_gamut(encoded))
        self.assertGreater(encoded[0], 1)
        self.assertLess(encoded[1], 0)

    def test_clipping_is_explicit_and_only_after_conversion(self):
        rgb = [[1, 0, 0], [-1, .5, .5]]
        result = convert_rgb(rgb, "display-p3", "srgb")
        clipped = convert_rgb(rgb, "display-p3", "srgb", clip=True)
        np.testing.assert_array_equal(clipped, np.clip(result, 0, 1))
        preclipped = convert_rgb(np.clip(rgb, 0, 1), "display-p3", "srgb", clip=True)
        self.assertGreater(np.max(np.abs(preclipped - clipped)), .01)
        self.assertGreater(np.max(np.abs(convert_rgb(clipped, "srgb", "display-p3") - rgb)), .1)

    def test_transfer_and_matrix_order_is_not_interchangeable(self):
        rgb = np.array([.25, .5, .75])
        expected = fixture_convert(rgb, "display-p3", "srgb")
        correct = convert_rgb(rgb, "display-p3", "srgb")
        wrong = linear_rgb_matrix("display-p3-linear", "srgb-linear") @ rgb
        np.testing.assert_allclose(correct, expected, rtol=0, atol=2e-15)
        self.assertGreater(np.max(np.abs(correct - wrong)), .04)

    def test_conversion_and_synthetic_nonlinear_stage_do_not_commute(self):
        rgb = np.array([.2, .5, .8])
        # A toy square stage, not a claim about an App stage or its bindings.
        before = convert_rgb(rgb ** 2, "display-p3", "srgb")
        after = convert_rgb(rgb, "display-p3", "srgb") ** 2
        self.assertGreater(np.max(np.abs(before - after)), .05)

    def test_negative_and_extended_transfer_is_odd_and_unclipped(self):
        positive = np.array([[0, .001, .03], [.18, 1, 4]])
        for transfer in (srgb_eotf, srgb_oetf):
            np.testing.assert_array_equal(transfer(-positive), -transfer(positive))
            self.assertGreater(transfer(positive)[1, 2], 1)
        self.assertTrue(np.signbit(srgb_eotf([-0.0, 0, 0])[0]))
        self.assertTrue(np.signbit(srgb_oetf([-0.0, 0, 0])[0]))

    def test_large_finite_values_and_underflow_policy(self):
        linear = np.array([1e300, -1e300, 1e299])
        encoded = srgb_oetf(linear)
        np.testing.assert_allclose(srgb_eotf(encoded), linear, rtol=3e-13, atol=0)
        converted = convert_rgb(linear, "display-p3-linear", "srgb-linear")
        np.testing.assert_allclose(convert_rgb(converted, "srgb-linear", "display-p3-linear"),
                                   linear, rtol=1e-14, atol=0)
        tiny = np.nextafter(0.0, 1.0)
        with np.errstate(all="raise"):
            np.testing.assert_array_equal(srgb_eotf([tiny, -tiny, 0]), [0, -0.0, 0])
            self.assertTrue(np.all(np.isfinite(srgb_oetf([tiny, -tiny, 0]))))
            self.assertTrue(np.all(np.isfinite(srgb_oetf([np.finfo(float).max, 0, 0]))))

    def test_unrepresentable_results_raise_even_when_clipping(self):
        largest = np.finfo(np.float64).max
        for clip in (False, True):
            with self.assertRaises(ValueError):
                convert_rgb([largest, 0, 0], "srgb", "srgb-linear", clip=clip)
            with self.assertRaises(ValueError):
                convert_rgb([largest, 0, 0], "display-p3-linear", "srgb-linear", clip=clip)
        with self.assertRaises(ValueError):
            srgb_eotf([-largest, 0, 0])
        # No unnecessary powers or intermediate XYZ calculation for identity.
        np.testing.assert_array_equal(convert_rgb([largest, 0, 0], "srgb", "srgb"),
                                      [largest, 0, 0])

    def test_batch_shape_dtype_noncontiguous_and_empty(self):
        rgb = (np.arange(72, dtype=np.float32).reshape(3, 8, 3) / 72)[:, ::2, :]
        self.assertFalse(rgb.flags.c_contiguous)
        for shape in ((3,), (2, 3), (2, 4, 3), (0, 3), (2, 0, 3)):
            result = convert_rgb(np.zeros(shape), "srgb", "display-p3")
            self.assertEqual(result.shape, shape)
            self.assertEqual(result.dtype, np.dtype("float64"))
            self.assertEqual(out_of_gamut(result).shape, shape[:-1])
        actual = convert_rgb(rgb, "srgb", "display-p3")
        for index in np.ndindex(rgb.shape[:-1]):
            np.testing.assert_allclose(actual[index], fixture_convert(rgb[index], "srgb", "display-p3"),
                                       rtol=0, atol=3e-15)

    def test_integer_values_are_not_implicitly_normalized(self):
        np.testing.assert_array_equal(convert_rgb([255, 0, 1], "srgb", "srgb"), [255, 0, 1])
        self.assertTrue(out_of_gamut([255, 0, 1]))

    def test_input_not_mutated_and_returned_matrices_are_independent(self):
        rgb = np.array([[.1, .2, .3], [-.2, 1.1, .4]])
        original = rgb.copy()
        rgb.setflags(write=False)
        for target in COLOR_SPACES:
            result = convert_rgb(rgb, "srgb", target, clip=True)
            self.assertFalse(np.shares_memory(rgb, result))
        np.testing.assert_array_equal(rgb, original)
        matrix = linear_rgb_to_xyz_matrix("srgb-linear")
        matrix[:] = 99
        np.testing.assert_allclose(linear_rgb_to_xyz_matrix("srgb-linear"), W3C_SRGB_TO_XYZ,
                                   rtol=0, atol=4e-16)
        matrix = linear_rgb_matrix("srgb-linear", "display-p3-linear")
        matrix[:] = 99
        self.assertLess(linear_rgb_matrix("srgb-linear", "display-p3-linear").max(), 2)

    def test_out_of_gamut_per_triplet_with_explicit_tolerance(self):
        values = [[0, .5, 1], [-1e-14, .5, 1], [0, .5, 1 + 1e-14], [2, -.2, 0]]
        np.testing.assert_array_equal(out_of_gamut(values), [False, True, True, True])
        np.testing.assert_array_equal(out_of_gamut(values, tolerance=1e-12),
                                      [False, False, False, True])
        for tolerance in (-1, np.nan, np.inf, "0", None, True, [0], complex(0)):
            with self.subTest(tolerance=tolerance), self.assertRaises(ValueError):
                out_of_gamut([0, 0, 0], tolerance=tolerance)

    def test_invalid_modes_and_implicit_encoded_matrix_rejected(self):
        invalid = ("DisplayP3", "p3", "linear-srgb", "dci-p3", "gamma2.6", "", None, 1, [], {})
        for mode in invalid:
            for source, target in ((mode, "srgb"), ("srgb", mode)):
                with self.subTest(source=source, target=target), self.assertRaises(ValueError):
                    convert_rgb([0, 0, 0], source, target)
        for mode in ("srgb", "display-p3", "bogus"):
            with self.assertRaises(ValueError):
                linear_rgb_to_xyz_matrix(mode)
            with self.assertRaises(ValueError):
                linear_rgb_matrix(mode, "srgb-linear")
            with self.assertRaises(ValueError):
                linear_rgb_matrix("srgb-linear", mode)
        with self.assertRaises(TypeError):
            convert_rgb([0, 0, 0])

    def test_invalid_shape_type_finiteness_and_clip_rejected(self):
        invalid = [0, [], [0, 0], [0, 0, 0, 1], np.zeros((3, 2)),
                   [[0, 0, 0], [1, 2]], [np.nan, 0, 0], [np.inf, 0, 0],
                   [-np.inf, 0, 0], [1 + 2j, 0, 0], ["0", "1", "0"],
                   [False, True, False], np.array([0, 0, 0], dtype=object)]
        operations = [srgb_eotf, srgb_oetf, out_of_gamut,
                      lambda x: convert_rgb(x, "srgb", "display-p3"),
                      lambda x: convert_rgb(x, "srgb", "srgb")]
        for value, operation in itertools.product(invalid, operations):
            with self.subTest(value=repr(value), operation=operation), self.assertRaises(ValueError):
                operation(value)
        for clip in ("false", 0, 1, None, []):
            with self.subTest(clip=clip), self.assertRaises(ValueError):
                convert_rgb([0, 0, 0], "srgb", "srgb", clip=clip)


if __name__ == "__main__":
    unittest.main(verbosity=2)
