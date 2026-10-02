"""Keyless independent-oracle tests; integration evaluates real recovered resources."""
from __future__ import annotations
import itertools
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from experiments.validation import validate_look_families as f


def affine_table(n, matrix=None, offset=None):
    grid=np.arange(n,dtype=float)/(n-1)
    b,g,r=np.meshgrid(grid,grid,grid,indexing='ij')
    values=np.stack([r,g,b],axis=-1)
    if matrix is not None:values=values@np.asarray(matrix).T
    if offset is not None:values=values+offset
    return values


def scalar_texture(table, point):
    n=len(table)
    axis=[]
    for v in point:
        q=max(0.,min(n-1.,v*n-.5));lo=int(q)
        axis.append([(lo,1-(q-lo)),(min(lo+1,n-1),q-lo)])
    result=[0.,0.,0.]
    for red,green,blue in itertools.product(*axis):
        for channel in range(3):
            result[channel]+=red[1]*green[1]*blue[1]*float(table[blue[0],green[0],red[0],channel])
    return result


def binding(recipe='source_to_primary',space='srgb'):
    return {'recipe':recipe,'input_space':space,'output_space':space,
            'primary_cube':{'filename':'p'},'secondary_cube':{'filename':'s'}}


class SamplerOracleTests(unittest.TestCase):
    def test_all_texel_centers_recover_nodes(self):
        table=affine_table(4,[[.1,.2,.3],[.4,.3,.1],[.5,.1,.2]],[.02,.03,.04])
        centers=(np.arange(4)+.5)/4
        xyz=np.array(list(itertools.product(centers,repeat=3)))
        expected=np.array([table[b,g,r] for r,g,b in itertools.product(range(4),repeat=3)])
        np.testing.assert_array_equal(f.texture(table,xyz),expected)

    def test_clamp_to_edge_extended_coordinates(self):
        table=affine_table(4)
        np.testing.assert_array_equal(f.texture(table,[[-5,.5,8]]),[[0,.5,1]])

    def test_r_fast_axis_order_not_transposed(self):
        table=affine_table(3)
        np.testing.assert_array_equal(f.texture(table,[[1,0,0],[0,1,0],[0,0,1]]),np.eye(3))

    def test_affine_function_matches_analytical_coordinates(self):
        n=7;matrix=np.array([[.1,.3,.4],[.2,-.1,.7],[.4,.2,.1]]);offset=np.array([.2,.1,-.3])
        x=np.random.default_rng(7).uniform(-.2,1.2,(300,3))
        wanted=np.clip(x*n-.5,0,n-1)/(n-1)@matrix.T+offset
        np.testing.assert_allclose(f.texture(affine_table(n,matrix,offset),x),wanted,atol=3e-16,rtol=0)

    def test_vectorized_matches_independent_scalar_loops(self):
        rng=np.random.default_rng(4);table=rng.random((5,5,5,3));x=rng.uniform(-.5,1.5,(97,3))
        np.testing.assert_allclose(f.texture(table,x),[scalar_texture(table,p) for p in x],atol=3e-16,rtol=0)

    def test_conventional_cube_sampling_is_detectably_different(self):
        x=np.array([[.2,.4,.8]])
        self.assertGreater(float(np.max(np.abs(f.texture(affine_table(4),x)-x))),.05)

    def test_nonfinite_coordinates_rejected(self):
        with self.assertRaises(ValueError):f.texture(affine_table(2),[[float('nan'),0,0]])

    def test_input_is_not_modified(self):
        x=np.array([[-.1,.4,1.2]]);old=x.copy();f.texture(affine_table(3),x)
        np.testing.assert_array_equal(x,old)


class DomainAndCompositionTests(unittest.TestCase):
    def setUp(self):
        self.x=np.random.default_rng(6).random((100,3))
        self.tables={'p':affine_table(3,np.diag([.8,.7,.6]),[.02,.03,.04]),
                     's':affine_table(3,np.diag([.4,.5,.6]),[.1,.2,.3]),
                     'filter':affine_table(3,[[.2,.5,.3]]*3)}
        self.attachment={'cube':{'filename':'filter'}}

    def test_both_primaries_have_d65_white(self):
        np.testing.assert_allclose(f.SRGB_XYZ.sum(axis=1),f.P3_XYZ.sum(axis=1),atol=3e-16,rtol=0)

    def test_extended_roundtrip_retains_gamut(self):
        x=np.array([[-.1,.5,1.2],[.95,.04,.2]])
        back=f.convert(f.convert(x,'srgb','display-p3'),'display-p3','srgb')
        np.testing.assert_allclose(back,x,atol=2e-14,rtol=0)
        self.assertLess(f.convert([[0,1,0]],'display-p3','srgb')[0,0],0)

    def test_p3_conversion_not_numeric_identity(self):
        self.assertGreater(float(np.max(np.abs(f.convert([[1,0,0]],'srgb','display-p3')-[1,0,0]))),.1)

    def test_transfer_standard_breakpoints_and_negative_sign(self):
        np.testing.assert_allclose(f.transfer([0,.04045,1],True),[0,.04045/12.92,1])
        np.testing.assert_array_equal(f.transfer([-1,-.02,0],True),-f.transfer([1,.02,0],True))

    def test_identity_is_independent_copy(self):
        got=f.compose(self.x,binding('identity'),{},100)
        np.testing.assert_array_equal(got,self.x);self.assertFalse(np.shares_memory(got,self.x))

    def test_single_zero_equals_original_even_with_prefilter(self):
        np.testing.assert_array_equal(f.compose(self.x,binding(),self.tables,0,self.attachment),self.x)

    def test_single_full_strength_matches_single_table(self):
        np.testing.assert_array_equal(f.compose(self.x,binding(),self.tables,100),f.texture(self.tables['p'],self.x))

    def test_single_fractional_strength_uses_original(self):
        y=f.texture(self.tables['p'],self.x)
        np.testing.assert_allclose(f.compose(self.x,binding(),self.tables,37.25),self.x*.6275+y*.3725,atol=2e-16)

    def test_dual_zero_and_hundred_are_endpoints_not_original(self):
        b=binding('secondary_to_primary')
        np.testing.assert_array_equal(f.compose(self.x,b,self.tables,0),f.texture(self.tables['s'],self.x))
        np.testing.assert_array_equal(f.compose(self.x,b,self.tables,100),f.texture(self.tables['p'],self.x))
        self.assertGreater(float(np.max(np.abs(f.compose(self.x,b,self.tables,0)-self.x))),.1)

    def test_dual_midpoint(self):
        b=binding('secondary_to_primary');a=f.compose(self.x,b,self.tables,0);z=f.compose(self.x,b,self.tables,100)
        np.testing.assert_allclose(f.compose(self.x,b,self.tables,50),(a+z)/2,atol=2e-16)

    def test_prefilter_precedes_tone(self):
        correct=f.texture(self.tables['p'],f.texture(self.tables['filter'],self.x))
        got=f.compose(self.x,binding(),self.tables,100,self.attachment)
        np.testing.assert_array_equal(got,correct)
        wrong=f.texture(self.tables['filter'],f.texture(self.tables['p'],self.x))
        self.assertGreater(float(np.max(np.abs(got-wrong))),.01)

    def test_prefilter_does_not_replace_source_blend_operand(self):
        y=f.compose(self.x,binding(),self.tables,100,self.attachment)
        got=f.compose(self.x,binding(),self.tables,50,self.attachment)
        np.testing.assert_allclose(got,.5*self.x+.5*y,atol=2e-16)
        wrong=.5*f.texture(self.tables['filter'],self.x)+.5*y
        self.assertGreater(float(np.max(np.abs(got-wrong))),.1)

    def test_p3_dual_mixes_before_conversion(self):
        b=binding('secondary_to_primary','display-p3');x=f.convert(self.x,'srgb','display-p3')
        before=(f.texture(self.tables['p'],x)+f.texture(self.tables['s'],x))/2
        wanted=f.convert(before,'display-p3','srgb')
        np.testing.assert_allclose(f.compose(self.x,b,self.tables,50),wanted,atol=2e-16)

    def test_final_quantizer_clips_and_rounds_half_up(self):
        np.testing.assert_array_equal(f.quantize([[-.1,.5/255,1.1],[.5,1,0]]),[[0,1,255],[128,255,0]])

    def test_original_catalog_has_no_strength_slider(self):
        raw=json.loads(f.CATALOG.read_text());original=next(x for x in raw['looks'] if x['id']=='original')
        self.assertFalse(original['adjustable']);self.assertEqual(original['default_strength'],100)

    def test_catalog_is_four_execution_families_plus_original(self):
        raw=json.loads(f.CATALOG.read_text());counts={}
        for row in raw['looks']:counts[f.family(row)]=counts.get(f.family(row),0)+1
        self.assertEqual(counts,{'identity':1,'single-p3':3,'single-srgb':16,'dual-srgb':1,'dual-p3':1})


class ImageSamplingTests(unittest.TestCase):
    def test_metadata_less_preview_requires_explicit_orientation(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'preview.png'
            Image.fromarray(np.zeros((4,8,3),dtype=np.uint8)).save(path)
            _,raw=f.sample_image(path)
            _,upright=f.sample_image(path,orientation_override=6)
            self.assertEqual(raw['upright_size'],[8,4])
            self.assertEqual(upright['upright_size'],[4,8])
            self.assertEqual(upright['orientation_override'],6)

    def test_unexpected_orientation_override_rejected(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'preview.png'
            Image.fromarray(np.zeros((4,8,3),dtype=np.uint8)).save(path)
            with self.assertRaises(ValueError):f.sample_image(path,orientation_override=8)


class ExternalFixtureConfigurationTests(unittest.TestCase):
    def test_no_manifest_needs_no_photo(self):
        self.assertEqual(f.load_photo_fixtures(None), ([], []))

    def test_manifest_resolves_relative_paths_without_opening_photos(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root/'fixtures.json'
            manifest.write_text(json.dumps({
                'images': [{'path': 'private-input.jpg', 'orientation_override': 6}],
                'anchors': [{'look': 'vivid', 'strength': 100,
                             'source': 'private-input.jpg', 'reference': 'reference.jpg'}],
            }))
            images, anchors = f.load_photo_fixtures(manifest)
            self.assertEqual(images, [(root/'private-input.jpg', 6)])
            self.assertEqual(anchors, [('vivid', 100, root/'private-input.jpg', root/'reference.jpg', None)])
            self.assertFalse((root/'private-input.jpg').exists())

    def test_invalid_manifest_schema_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp)/'fixtures.json'
            manifest.write_text('{"undeclared": []}')
            with self.assertRaises(ValueError):
                f.load_photo_fixtures(manifest)

    def test_nonfinite_anchor_strength_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp)/'fixtures.json'
            manifest.write_text(json.dumps({'anchors': [
                {'look': 'vivid', 'strength': 'nan', 'source': 'a.jpg', 'reference': 'b.jpg'}]}))
            with self.assertRaises(ValueError):
                f.load_photo_fixtures(manifest)


class ResourceParsingTests(unittest.TestCase):
    def test_half_restore_and_r_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'test.cube';v=affine_table(2)*.123456789
            path.write_text('TITLE "test"\nLUT_3D_SIZE 2\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n'+'\n'.join(' '.join(map(str,row)) for row in v.reshape(-1,3)))
            np.testing.assert_array_equal(f.load_cube(path),v)
            np.testing.assert_array_equal(f.load_cube(path,True),v.astype(np.float16).astype(float))

    def test_duplicate_dimensions_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.cube';path.write_text('LUT_3D_SIZE 2\nLUT_3D_SIZE 2\n')
            with self.assertRaises(ValueError):f.load_cube(path)

    def test_non_unit_domain_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.cube';path.write_text('LUT_3D_SIZE 2\nDOMAIN_MAX 2 2 2\n')
            with self.assertRaises(ValueError):f.load_cube(path)


if __name__=='__main__':unittest.main()
