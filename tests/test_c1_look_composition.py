"""Float composition tests use independent corner sums and standard matrices."""
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import unittest

import numpy as np

from apps.local_looks.catalog import LOOK_BINDINGS, COLOR_FILTERS
from experiments.c1 import look_composition as c
from experiments.validation import validate_look_families as oracle
from reproduction.ios_looks.renderer import TextureLUT


def binding(recipe='source_to_primary', space='srgb', **overrides):
    row = {'id': 'toy', 'group': 'color', 'recipe': recipe,
           'input_space': space, 'output_space': space,
           'primary_cube': {'filename': 'p.cube'},
           'secondary_cube': {'filename': 's.cube'} if recipe == 'secondary_to_primary' else None,
           'filter_options': ['red']}
    return dict(row, **overrides)


def constant(value, n=2):
    return TextureLUT(np.broadcast_to(np.asarray(value), (n, n, n, 3)).copy())


def affine_table(matrix, offset=0, n=3):
    axis = np.linspace(0, 1, n)
    b, g, r = np.meshgrid(axis, axis, axis, indexing='ij')
    values = np.stack([r, g, b], axis=-1) @ np.asarray(matrix).T + offset
    return TextureLUT(values)


def write_cube(path, values, restore=False):
    n = len(values)
    lines = [f'LUT_3D_SIZE {n}'] + [' '.join(f'{v:.17g}' for v in row) for row in values.reshape(-1, 3)]
    path.write_text('\n'.join(lines) + '\n')
    return {'filename': path.name, 'grid_size': n, 'restore_half': restore,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


class FloatCompositionTests(unittest.TestCase):
    def test_single_full_strength_matches_independent_corner_sum(self):
        p = affine_table([[.7, .1, 0], [.2, .5, .1], [0, .2, .6]], .03)
        x = np.random.default_rng(123).uniform(-.2, 1.2, (101, 3))
        result = c.compose_look(x, binding(), primary=p)
        np.testing.assert_allclose(result, oracle.texture(p.values, x), atol=2e-16)

    def test_fractional_source_blend_uses_original_and_no_quantizer(self):
        x = np.array([[.123456789, .876543211, .001]])
        p = constant([.712345678, .023456789, .056789012])
        result = c.compose_look(x, binding(), primary=p, strength=37.25)
        np.testing.assert_allclose(result, x + (p.values[0, 0, 0] - x) * .3725, atol=1e-16)
        self.assertGreater(np.max(np.abs(result * 255 - np.round(result * 255))), .01)

    def test_no_delta_e_cap_or_gray_suppression(self):
        x = np.array([[1., 0, 0], [0, 1., 0], [0, 0, 1.]])
        gray = constant([.4, .4, .4])
        np.testing.assert_array_equal(c.compose_look(x, binding(), primary=gray), np.full_like(x, .4))
        tone = constant([.44, .4, .37])
        np.testing.assert_array_equal(c.compose_look(x, binding(), primary=tone),
                                      np.broadcast_to([.44, .4, .37], x.shape))

    def test_single_zero_exact_original_with_prefilter(self):
        x = np.array([[-.2, .1234, 1.2]])
        y = c.compose_look(x, binding(), primary=constant([0, 0, 0]),
                           prefilter=constant([1, 1, 1]), strength=0)
        np.testing.assert_array_equal(y, x)
        self.assertFalse(np.shares_memory(x, y))

    def test_dual_zero_and_hundred_are_low_and_high(self):
        x = np.array([[.2, .3, .4]])
        p, s = constant([.8, .7, .6]), constant([.1, .2, .3])
        for strength, expected in [(0, [.1, .2, .3]), (100, [.8, .7, .6]), (25, [.275, .325, .375])]:
            y = c.compose_look(x, binding('secondary_to_primary'), primary=p, secondary=s, strength=strength)
            np.testing.assert_allclose(y, [expected], atol=2e-16)

    def test_prefilter_runs_before_both_tones(self):
        x = np.random.default_rng(13).random((87, 3))
        f = affine_table([[0, 0, .6], [.7, 0, 0], [0, .8, 0]], .05)
        p = affine_table([[.8, 0, 0], [0, .4, 0], [0, 0, .7]], .07)
        s = affine_table([[.3, .2, 0], [0, .2, .1], [.1, 0, .2]], .13)
        row = binding('secondary_to_primary')
        y = c.compose_look(x, row, primary=p, secondary=s, prefilter=f, strength=37.25)
        coords = oracle.texture(f.values, x)
        expected = .6275 * oracle.texture(s.values, coords) + .3725 * oracle.texture(p.values, coords)
        np.testing.assert_allclose(y, expected, atol=3e-16)
        wrong = oracle.texture(f.values, .6275 * oracle.texture(s.values, x) + .3725 * oracle.texture(p.values, x))
        self.assertGreater(np.max(np.abs(y - wrong)), .02)

    def test_prefilter_does_not_replace_single_blend_operand(self):
        x = np.array([[.3, .6, .1]])
        p = affine_table(np.eye(3) * .7, .05)
        f = constant([.8, .2, .4])
        y = c.compose_look(x, binding(), primary=p, prefilter=f, strength=25)
        expected = .75 * x + .25 * oracle.texture(p.values, [[.8, .2, .4]])
        np.testing.assert_allclose(y, expected, atol=2e-16)

    def test_p3_input_and_output_use_independent_standard_matrices(self):
        x = np.random.default_rng(19).random((131, 3))
        p = affine_table([[.6, .2, 0], [.1, .7, .1], [0, .2, .7]], .02)
        y = c.compose_look(x, binding(space='display-p3'), primary=p, strength=75)
        working = oracle.convert(x, 'srgb', 'display-p3')
        full = oracle.convert(oracle.texture(p.values, working), 'display-p3', 'srgb')
        np.testing.assert_allclose(y, .25*x + .75*full, atol=6e-15)

    def test_p3_dual_mixes_before_output_conversion(self):
        x = np.array([[.2, .3, .4]])
        p, s = constant([.85, .1, .1]), constant([.05, .75, .2])
        y = c.compose_look(x, binding('secondary_to_primary', 'display-p3'),
                           primary=p, secondary=s, strength=25)
        expected = oracle.convert(.75*s.values[0, 0, 0] + .25*p.values[0, 0, 0], 'display-p3', 'srgb')
        np.testing.assert_allclose(y, expected[None], atol=6e-15)
        wrong = .75*oracle.convert(s.values[0, 0, 0], 'display-p3', 'srgb') + .25*oracle.convert(p.values[0, 0, 0], 'display-p3', 'srgb')
        self.assertGreater(np.max(np.abs(y - wrong)), .01)

    def test_p3_extended_output_preserved(self):
        p = constant([1., 0, 0])
        y = c.compose_look([.5, .5, .5], binding(space='display-p3'), primary=p)
        self.assertGreater(y[0], 1)
        self.assertLess(y[1], 0)

    def test_shape_dtype_and_nonmutation(self):
        x = np.random.default_rng(21).random((2, 4, 3)).astype(np.float32)
        before = x.copy()
        y = c.compose_look(x, binding(), primary=constant([.2, .3, .4]), strength=25)
        self.assertEqual(y.shape, x.shape)
        self.assertEqual(y.dtype, np.float64)
        self.assertFalse(np.shares_memory(y, x))
        np.testing.assert_array_equal(x, before)
        self.assertEqual(c.compose_look(np.empty((0, 3)), binding(), primary=constant([0, 0, 0])).shape, (0, 3))

    def test_original_identity_and_no_strength(self):
        row = binding('identity', primary_cube=None, filter_options=[])
        x = np.array([.2, .3, .4])
        y = c.compose_look(x, row)
        np.testing.assert_array_equal(x, y)
        self.assertFalse(np.shares_memory(x, y))
        with self.assertRaises(ValueError): c.compose_look(x, row, strength=0)
        with self.assertRaises(ValueError): c.compose_look(x, row, primary=constant([0, 0, 0]))

    def test_invalid_rgb_rejected(self):
        for x in ([1, 2], [1, 2, 3, 4], ['1', '2', '3'], [True, False, True], [1j, 0, 0], [np.inf, 0, 0], [np.nan, 0, 0]):
            with self.subTest(x=x), self.assertRaises(ValueError):
                c.compose_look(x, binding(), primary=constant([0, 0, 0]))

    def test_invalid_strength_rejected(self):
        for strength in (-.1, 100.1, np.inf, np.nan, True, '50', [50]):
            with self.subTest(strength=strength), self.assertRaises(ValueError):
                c.compose_look([0, 0, 0], binding(), primary=constant([0, 0, 0]), strength=strength)

    def test_recipe_table_requirements(self):
        x = [0, 0, 0]
        p = constant([0, 0, 0])
        with self.assertRaises(ValueError): c.compose_look(x, binding(recipe='other'), primary=p)
        with self.assertRaises(ValueError): c.compose_look(x, binding(input_space='xyz'), primary=p)
        with self.assertRaises(TypeError): c.compose_look(x, binding())
        with self.assertRaises(TypeError): c.compose_look(x, binding('secondary_to_primary'), primary=p)
        with self.assertRaises(ValueError): c.compose_look(x, binding(), primary=p, secondary=p)
        with self.assertRaises(TypeError): c.compose_look(x, binding(), primary=p, prefilter=np.zeros(3))


class CatalogLoaderTests(unittest.TestCase):
    def test_same_filename_preserves_per_binding_precision_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'same.cube'
            a = np.full((2, 2, 2, 3), .123456789)
            exact_spec = write_cube(path, a)
            half_spec = dict(exact_spec, restore_half=True)
            rows = {'exact': binding(id='exact', primary_cube=exact_spec),
                    'half': binding(id='half', primary_cube=half_spec)}
            loader = c.FloatLookCompositor(directory, bindings=rows)
            x = np.array([[.25, .5, .75]])
            np.testing.assert_array_equal(loader.apply(x, 'exact'), np.full_like(x, .123456789))
            np.testing.assert_array_equal(loader.apply(x, 'half'), np.full_like(x, float(np.float16(.123456789))))
            self.assertEqual(len(loader._cache), 2)

    def test_hash_grid_and_path_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = write_cube(Path(directory)/'p.cube', constant([.2, .3, .4]).values)
            for wrong in (dict(spec, sha256='0'*64), dict(spec, grid_size=3),
                          dict(spec, filename='../p.cube'), dict(spec, filename='C:\\p.cube'),
                          dict(spec, restore_half=1)):
                loader = c.FloatLookCompositor(directory, bindings={'toy': binding(primary_cube=wrong)})
                with self.assertRaises(ValueError): loader.apply([.1, .2, .3], 'toy')

    def test_zero_bypass_and_unknown_filter_validate_without_files(self):
        loader = c.FloatLookCompositor('/nonexistent-resource-directory')
        x = np.array([[-.1, .3, 1.1]])
        np.testing.assert_array_equal(loader.apply(x, 'vivid', 0), x)
        np.testing.assert_array_equal(loader.apply(x, 'original'), x)
        with self.assertRaises(ValueError): loader.apply(x, 'unknown')
        with self.assertRaises(ValueError): loader.apply(x, 'vivid', 0, 'red')

    def test_cache_does_not_reopen_same_table(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'p.cube'
            spec = write_cube(path, constant([.2, .3, .4]).values)
            loader = c.FloatLookCompositor(directory, bindings={'toy': binding(primary_cube=spec)})
            first = loader.apply([.5, .5, .5], 'toy')
            path.unlink()
            np.testing.assert_array_equal(loader.apply([.5, .5, .5], 'toy'), first)


class PlanTests(unittest.TestCase):
    def test_default_count_strengths_and_groups(self):
        plan = c.build_variant_plan()
        self.assertEqual(len(plan), 206)
        main = [v for v in plan if v.color_filter is None]
        attached = [v for v in plan if v.color_filter is not None]
        self.assertEqual(len(main), 86)
        self.assertEqual(len(attached), 120)
        self.assertEqual(Counter(v.group for v in main), {'color': 52, 'monochrome': 24, 'artist': 10})
        self.assertEqual({v.look_id for v in plan}, set(LOOK_BINDINGS)-{'original'})
        self.assertEqual({v.look_id for v in plan if v.strength == 0}, {'steve', 'greg'})
        self.assertEqual({v.strength for v in attached}, {25, 50, 75, 100})
        self.assertTrue(all(v.group == 'monochrome' and v.look_id != 'greg' for v in attached))
        self.assertEqual({v.color_filter for v in attached}, set(COLOR_FILTERS))

    def test_stable_ascii_paths_and_unique_descriptions(self):
        first = c.build_variant_plan()
        self.assertEqual(first, c.build_variant_plan())
        self.assertEqual(len({v.relative_path.casefold() for v in first}), 206)
        self.assertEqual(len({v.description.casefold() for v in first}), 206)
        for v in first:
            self.assertTrue(v.relative_path.isascii() and v.description.isascii())
            path = PurePosixPath(v.relative_path)
            self.assertFalse(path.is_absolute())
            self.assertNotIn('..', path.parts)
            self.assertTrue(path.name.startswith('LeicaQTyp116-'))
            self.assertEqual(path.suffix, '.icm')
            self.assertIn(f'S{v.strength:03d}', v.description)
            if v.color_filter:
                self.assertIn('Filter'+v.color_filter.title(), v.description)
            self.assertLess(len(path.name), 160)

    def test_json_plan_contains_no_paths_to_local_inputs(self):
        data = [v.to_dict() for v in c.build_variant_plan()]
        text = json.dumps(data)
        self.assertEqual(len(json.loads(text)), 206)
        self.assertNotIn('/home/', text)
        self.assertNotIn('official_description', text)

    def test_invalid_id_and_casefold_collision_rejected(self):
        for bad_id in ('../bad', 'éclair', 'Bad', 'a-b'):
            with self.assertRaises(ValueError): c.build_variant_plan({bad_id: binding(id=bad_id, filter_options=[])})
        rows = {'foo_bar': binding(id='foo_bar', filter_options=[]),
                'foobar': binding(id='foobar', filter_options=[])}
        with self.assertRaises(ValueError): c.build_variant_plan(rows)

    def test_invalid_group_recipe_and_mapping_key_rejected(self):
        for row in (binding(group='unknown'), binding(recipe='other'), binding(id='different')):
            with self.assertRaises(ValueError): c.build_variant_plan({'toy': row})


@unittest.skipUnless(c.DEFAULT_RESOURCE_DIR.is_dir(), 'Prepared CUBE resources absent')
class PreparedResourcesTests(unittest.TestCase):
    def test_all_206_variants_against_independent_float_oracle(self):
        loader = c.FloatLookCompositor()
        tables = {}
        for row in list(LOOK_BINDINGS.values()) + list(COLOR_FILTERS.values()):
            specs = [row['cube']] if 'cube' in row else [row['primary_cube'], row['secondary_cube']]
            for spec in specs:
                if spec is not None and spec['filename'] not in tables:
                    tables[spec['filename']] = oracle.load_cube(c.DEFAULT_RESOURCE_DIR/spec['filename'], spec['restore_half'])
        rng = np.random.default_rng(2611)
        x = np.vstack([np.eye(3), np.zeros((1, 3)), np.ones((1, 3)), rng.random((89, 3))])
        original = x.copy()
        for variant in c.build_variant_plan():
            with self.subTest(path=variant.relative_path):
                row = LOOK_BINDINGS[variant.look_id]
                f = None if variant.color_filter is None else COLOR_FILTERS[variant.color_filter]
                expected = oracle.compose(x, row, tables, variant.strength, f)
                actual = loader.apply_variant(x, variant)
                np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=0)
                self.assertEqual(actual.dtype, np.float64)
                self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_array_equal(x, original)
        self.assertEqual(len(loader._cache), 28)


if __name__ == '__main__':
    unittest.main(verbosity=2)
