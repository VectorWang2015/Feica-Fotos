#!/usr/bin/env python3
"""Synthetic tests for the independent color wrapper and 16-bit file writer."""
import hashlib
from pathlib import Path
import struct
import tempfile
import unittest
import numpy as np
from PIL import ImageCms
import os
if __package__:
    from .color_math import Cube,LCMS,TYPE_LAB_DBL,encode_lab_v2
    from .build_rendered_icc import profile_bytes
    from .validate_rendered_icc import de2000,structure
    from .tiff16_io import read_rgb16,write_rgb16
else:
    from color_math import Cube,LCMS,TYPE_LAB_DBL,encode_lab_v2
    from build_rendered_icc import profile_bytes
    from validate_rendered_icc import de2000,structure
    from tiff16_io import read_rgb16,write_rgb16


class Tests(unittest.TestCase):
    def test_cube_identity_axis_order(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'identity.cube'
            rows=['LUT_3D_SIZE 2']+[f'{r} {g} {b}' for b in [0,1] for g in [0,1] for r in [0,1]]
            p.write_text('\n'.join(rows)+'\n')
            x=np.random.default_rng(3).random((1000,3));cube=Cube(p)
            for method in ['tetrahedral','trilinear']:
                np.testing.assert_allclose(cube.evaluate(x,method),x,atol=1e-14)

    def test_cube_cross_channel_transform(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'swap.cube'
            rows=['LUT_3D_SIZE 2']+[f'{g} {b} {r}' for b in [0,1] for g in [0,1] for r in [0,1]]
            p.write_text('\n'.join(rows)+'\n')
            x=np.random.default_rng(8).random((100,3))
            np.testing.assert_allclose(Cube(p).evaluate(x),x[:,[1,2,0]],atol=1e-14)

    def test_lab_v2_known_encoding(self):
        self.assertEqual(encode_lab_v2([0,0,0]).tolist(),[0,32768,32768])
        self.assertEqual(encode_lab_v2([100,0,0]).tolist(),[65280,32768,32768])
        with LCMS() as cms:
            for lab in [[0,0,0],[100,0,0],[53.5,-21.5,38.7]]:
                np.testing.assert_array_equal(encode_lab_v2(lab),cms.lab_v2_encode_one(lab))

    def test_standard_color_values(self):
        with LCMS() as cms:
            t=cms.transform(cms.srgb(),cms.lab(),output_format=TYPE_LAB_DBL)
            result=cms.apply(t,np.array([[1.,0,0],[0,1.,0],[0,0,1.]]))
        expected=np.array([[54.2896,80.8144,69.8897],[87.8194,-79.2749,80.9927],[29.5659,68.2862,-112.0329]])
        np.testing.assert_allclose(result,expected,atol=0.0002)

    def test_delta_e_standard_sample(self):
        x=np.array([[50,2.6772,-79.7751],[50,0,0]])
        y=np.array([[50,0,-82.7485],[50,0,0]])
        np.testing.assert_allclose(de2000(x,y),[2.0425,0],atol=0.0001)

    def test_icc_structure_and_identity(self):
        axis=np.linspace(0,1,17)
        x=np.stack(np.meshgrid(axis,axis,axis,indexing='ij'),axis=-1)
        with LCMS() as cms:
            srgb=cms.srgb();lab=cms.lab();t=cms.transform(srgb,lab,output_format=TYPE_LAB_DBL)
            raw=profile_bytes(cms.apply(t,x),'Synthetic identity input ICC')
            with tempfile.TemporaryDirectory() as d:
                p=Path(d)/'synthetic.icc';p.write_bytes(raw)
                self.assertTrue(structure(p)['tag_structure_valid'])
            to_srgb=cms.transform(cms.open(raw),srgb)
            samples=np.array([[0,0,0],[1,1,1],[1,0,0],[0,1,0],[0,0,1],[.5,.5,.5]],dtype=float)
            np.testing.assert_allclose(cms.apply(to_srgb,samples),samples,atol=0.001)

    def test_rgb16_tiff_exact_roundtrip(self):
        rng=np.random.default_rng(62)
        array=rng.integers(0,65536,size=(19,23,3),dtype=np.uint16)
        profile=ImageCms.ImageCmsProfile(ImageCms.createProfile('sRGB')).tobytes()
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'test.tif';write_rgb16(p,array,profile)
            actual,meta=read_rgb16(p)
            np.testing.assert_array_equal(array,actual)
            self.assertEqual(meta['icc'],profile)
            with self.assertRaises(FileExistsError):write_rgb16(p,array,profile)

    @unittest.skipUnless(os.environ.get('FEICA_RGB_VALIDATION_REPORT'),
                         'Optional archived validation report not supplied')
    def test_wrapper_random_accuracy_thresholds(self):
        import json
        p=Path(os.environ['FEICA_RGB_VALIDATION_REPORT'])
        data=json.loads(p.read_text())
        self.assertEqual(len(data['profiles']),5)
        for profile in data['profiles']:
            self.assertEqual(profile['grid'],65)
            g=profile['groups']['uniform_random']
            self.assertLess(g['rgb_code_error_8bit_equivalent']['mean_abs'],0.025)
            self.assertLess(g['rgb_code_error_8bit_equivalent']['p99_abs'],0.2)
            self.assertLess(g['deltaE00_D50']['p99'],0.05)


if __name__=='__main__':unittest.main(verbosity=2)
