#!/usr/bin/env python3
"""Synthetic mathematical tests, using no proprietary resource samples."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
if __package__:
    from .renderer import (TextureLUT, sample_texture, mix_tables, quantize_u8,
                           render_rgb8, VIVID, ETERNAL, STEVE1, STEVE3)
else:
    from renderer import (TextureLUT, sample_texture, mix_tables, quantize_u8,
                          render_rgb8, VIVID, ETERNAL, STEVE1, STEVE3)


def identity(n):
    b,g,r=np.meshgrid(np.linspace(0,1,n),np.linspace(0,1,n),np.linspace(0,1,n),indexing='ij')
    return TextureLUT(np.stack([r,g,b],axis=-1))


class SamplerTests(unittest.TestCase):
    def test_corners_and_channel_order(self):
        table=identity(5)
        colors=np.asarray([[0,0,0],[1,0,0],[0,1,0],[0,0,1],[1,1,1]],dtype=float)
        np.testing.assert_allclose(sample_texture(table,colors),colors,atol=1e-15)

    def test_texel_centers_return_exact_nodes(self):
        n=5;table=identity(n)
        rng=np.random.default_rng(703);i=rng.integers(0,n,(100,3))
        samples=sample_texture(table,(i+.5)/n)
        np.testing.assert_allclose(samples,i/(n-1),atol=1e-15)

    def test_identity_grid_is_not_identity_texture(self):
        n=17;x=np.asarray([[.1,.25,.75],[.02,.5,.98]])
        expected=np.clip((x*n-.5)/(n-1),0,1)
        np.testing.assert_allclose(sample_texture(identity(n),x),expected,atol=1e-15)
        self.assertGreater(np.max(np.abs(expected-x)),.01)

    def test_trilinear_multilinear_function(self):
        n=4;b,g,r=np.meshgrid(np.arange(n),np.arange(n),np.arange(n),indexing='ij')
        table=TextureLUT(np.stack([r+2*g+3*b,r*g,g*b],axis=-1))
        rng=np.random.default_rng(414);q=rng.uniform(0,n-1,(100,3));x=(q+.5)/n
        expected=np.stack([q[:,0]+2*q[:,1]+3*q[:,2],q[:,0]*q[:,1],q[:,1]*q[:,2]],axis=-1)
        np.testing.assert_allclose(sample_texture(table,x),expected,atol=1e-12)

    def test_clamp_edges(self):
        x=np.asarray([[-10,.5,10]])
        np.testing.assert_allclose(sample_texture(identity(5),x),[[0,.5,1]],atol=1e-15)

    def test_mix_is_between_tables_not_identity(self):
        a=TextureLUT(np.full((2,2,2,3),.2));b=TextureLUT(np.full((2,2,2,3),.8))
        x=np.asarray([[.1,.3,.7],[1,0,1]])
        np.testing.assert_allclose(mix_tables(a,b,x,0),.2)
        np.testing.assert_allclose(mix_tables(a,b,x,.5),.5)
        np.testing.assert_allclose(mix_tables(a,b,x,1),.8)

    def test_two_tables_can_have_different_sizes(self):
        a=TextureLUT(np.full((2,2,2,3),.1));b=TextureLUT(np.full((3,3,3,3),.9))
        np.testing.assert_allclose(mix_tables(a,b,np.asarray([[.25,.5,.75]]),.25),.3)

    def test_shapes_and_finiteness(self):
        x=np.zeros((3,4,3));self.assertEqual(sample_texture(identity(3),x).shape,x.shape)
        with self.assertRaises(ValueError):sample_texture(identity(3),[[0,0,float('nan')]])
        with self.assertRaises(ValueError):TextureLUT(np.zeros((2,3,2,3)))
        with self.assertRaises(ValueError):mix_tables(identity(2),identity(2),[[0,0,0]],2)

    def test_u8_round_and_clip(self):
        np.testing.assert_array_equal(quantize_u8(np.asarray([-1,0,.5,1,2])),[0,0,128,255,255])

    def test_input_and_table_not_mutated(self):
        t=identity(4);x=np.asarray([[.2,.3,.4]]);copy=x.copy();table=t.values.copy()
        sample_texture(t,x)
        np.testing.assert_array_equal(x,copy);np.testing.assert_array_equal(t.values,table)
        self.assertFalse(t.values.flags.writeable)


class ValidatedLookWrapperTests(unittest.TestCase):
    @staticmethod
    def write_constant_cube(directory, name, value):
        path = directory / name
        path.write_text('LUT_3D_SIZE 2\n' + (f'{value} {value} {value}\n' * 8))
        return path

    def test_vivid100_restores_binary16_before_existing_sampler(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            path = self.write_constant_cube(directory, VIVID, .29999)
            original = path.read_bytes()
            rgb = np.asarray([[[0, 100, 255], [127, 35, 208]]], dtype=np.uint8)
            parsed = TextureLUT.from_cube(path)
            recovered = TextureLUT(parsed.values.astype(np.float16).astype(np.float64))
            expected = quantize_u8(sample_texture(recovered, rgb.astype(np.float64)/255))
            result = render_rgb8(rgb, 'vivid', 100, directory, chunk_pixels=1)
            np.testing.assert_array_equal(result, expected)
            self.assertEqual(int(result[0, 0, 0]), 77)
            # Without the frozen half restoration, this synthetic tie-near value is76.
            self.assertEqual(int(quantize_u8(parsed.values)[0, 0, 0, 0]), 76)
            self.assertEqual(path.read_bytes(), original)

    def test_vivid_rejects_every_unvalidated_strength(self):
        rgb = np.zeros((1, 1, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            self.write_constant_cube(directory, VIVID, .29999)
            for strength in (0, 25, 50, 75, 99, 101, float('nan'), float('inf')):
                with self.subTest(strength=strength), self.assertRaisesRegex(ValueError, 'Only Vivid100'):
                    render_rgb8(rgb, 'vivid', strength, directory)

    def test_steve_eternal_keep_decimal_samples_and_previous_math(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            low = TextureLUT.from_cube(self.write_constant_cube(directory, STEVE1, .29999))
            high = TextureLUT.from_cube(self.write_constant_cube(directory, STEVE3, .70001))
            eternal = TextureLUT.from_cube(self.write_constant_cube(directory, ETERNAL, .29999))
            rgb = np.asarray([[[12, 127, 244], [0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
            normalized = rgb.astype(np.float64)/255
            for strength in (0, 25, 50, 75, 100):
                expected = quantize_u8(mix_tables(low, high, normalized, strength/100))
                np.testing.assert_array_equal(render_rgb8(rgb, 'steve', strength, directory), expected)
            result = render_rgb8(rgb, 'eternal', 100, directory)
            np.testing.assert_array_equal(result, quantize_u8(sample_texture(eternal, normalized)))
            self.assertEqual(int(result[0, 0, 0]), 76)  # No half restoration outside Vivid.


if __name__=='__main__':unittest.main(verbosity=2)
