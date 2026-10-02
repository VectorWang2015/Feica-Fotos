#!/usr/bin/env python3
"""Analytic checks for four Tone topologies; expectations do not call sampler."""
import unittest
import numpy as np
if __package__:
    from .renderer import TextureLUT, apply_tone_kernel, apply_tone_kernel_rgba
else:
    from renderer import TextureLUT, apply_tone_kernel, apply_tone_kernel_rgba


def table_of(function,n=5):
    axis=(np.arange(n,dtype=float)+.5)/n
    b,g,r=np.meshgrid(axis,axis,axis,indexing='ij')
    return TextureLUT(function(np.stack([r,g,b],axis=-1)))


def filter_formula(x):
    r,g,b=np.moveaxis(x,-1,0)
    return np.stack([.3+.4*r,.25+.5*g,.2+.4*b],axis=-1)


def primary_formula(x):
    r,g,b=np.moveaxis(x,-1,0)
    return np.stack([r*g,g+.1*b,.1+.8*r],axis=-1)


def secondary_formula(x):
    r,g,b=np.moveaxis(x,-1,0)
    return np.stack([.2+.3*b,r-.1*b,.3+.2*g],axis=-1)


class ToneCompositionTests(unittest.TestCase):
    def setUp(self):
        self.x=np.random.default_rng(771).uniform(.25,.75,(20,25,3))
        self.f=table_of(filter_formula,4)
        self.p=table_of(primary_formula,5)
        self.s=table_of(secondary_formula,7)

    def test_single_table_analytic(self):
        np.testing.assert_allclose(apply_tone_kernel(self.x,self.p),primary_formula(self.x),atol=2e-15)

    def test_two_tables_analytic_asymmetric_weight(self):
        expected=.7*secondary_formula(self.x)+.3*primary_formula(self.x)
        actual=apply_tone_kernel(self.x,self.p,secondary=self.s,blend_factor=.3)
        np.testing.assert_allclose(actual,expected,atol=2e-15)

    def test_color_filter_then_single_tone_analytic(self):
        actual=apply_tone_kernel(self.x,self.p,color_filter=self.f)
        expected=primary_formula(filter_formula(self.x))
        np.testing.assert_allclose(actual,expected,atol=2e-15)

    def test_color_filter_then_two_tones_analytic(self):
        c=filter_formula(self.x)
        expected=.35*secondary_formula(c)+.65*primary_formula(c)
        actual=apply_tone_kernel(self.x,self.p,secondary=self.s,blend_factor=.65,color_filter=self.f)
        np.testing.assert_allclose(actual,expected,atol=2e-15)

    def test_order_negative_control(self):
        right=primary_formula(filter_formula(self.x))
        wrong=filter_formula(primary_formula(self.x))
        self.assertGreater(np.max(np.abs(right-wrong)),.1)
        actual=apply_tone_kernel(self.x,self.p,color_filter=self.f)
        self.assertLess(np.max(np.abs(actual-right)),2e-15)

    def test_color_filter_is_shared_by_both_tone_coordinates(self):
        c=filter_formula(self.x);expected=.5*(secondary_formula(c)+primary_formula(c))
        wrong=.5*(secondary_formula(self.x)+primary_formula(c))
        actual=apply_tone_kernel(self.x,self.p,secondary=self.s,blend_factor=.5,color_filter=self.f)
        self.assertGreater(np.max(np.abs(expected-wrong)),.01)
        np.testing.assert_allclose(actual,expected,atol=2e-15)

    def test_factor_endpoints_are_tone_tables_not_original(self):
        c=filter_formula(self.x)
        for weight,expected in [(0,secondary_formula(c)),(1,primary_formula(c))]:
            y=apply_tone_kernel(self.x,self.p,secondary=self.s,blend_factor=weight,color_filter=self.f)
            np.testing.assert_allclose(y,expected,atol=2e-15)
        self.assertGreater(np.max(np.abs(secondary_formula(c)-self.x)),.1)

    def test_generic_shader_factor_not_silently_clamped(self):
        p=TextureLUT(np.full((2,2,2,3),2.));s=TextureLUT(np.full((3,3,3,3),-1.))
        np.testing.assert_allclose(apply_tone_kernel([[.2,.4,.6]],p,secondary=s,blend_factor=1.5),3.5)
        np.testing.assert_allclose(apply_tone_kernel([[.2,.4,.6]],p,secondary=s,blend_factor=-.5),-2.5)

    def test_filter_output_is_next_sampler_coordinate_not_clamped_output_color(self):
        f=TextureLUT(np.broadcast_to([-.2,.5,1.2],(2,2,2,3)))
        axis=np.linspace(0,1,3);b,g,r=np.meshgrid(axis,axis,axis,indexing='ij')
        p=TextureLUT(np.stack([r,g,b],axis=-1))
        np.testing.assert_allclose(apply_tone_kernel([[.4,.5,.6]],p,color_filter=f),[[0,.5,1]])

    def test_rgba_alpha_is_forced_one_without_unpremultiply(self):
        alpha=np.random.default_rng(99).uniform(0,1,(*self.x.shape[:-1],1))
        rgba=np.concatenate([self.x,alpha],axis=-1)
        out=apply_tone_kernel_rgba(rgba,self.p,color_filter=self.f)
        np.testing.assert_allclose(out[...,:3],primary_formula(filter_formula(self.x)),atol=2e-15)
        np.testing.assert_array_equal(out[...,3],1)

    def test_bad_parameter_combinations_rejected(self):
        with self.assertRaises(ValueError):apply_tone_kernel(self.x,self.p,blend_factor=.5)
        with self.assertRaises(ValueError):apply_tone_kernel(self.x,self.p,secondary=self.s)
        with self.assertRaises(ValueError):apply_tone_kernel(self.x,self.p,secondary=self.s,blend_factor=np.nan)
        with self.assertRaises(ValueError):apply_tone_kernel(self.x,self.p,secondary=self.s,blend_factor=[.5])
        with self.assertRaises(TypeError):apply_tone_kernel(self.x,None)
        with self.assertRaises(TypeError):apply_tone_kernel(self.x,self.p,color_filter='unknown')

    def test_shape_finiteness_and_immutability(self):
        source=self.x.copy();table=self.p.values.copy()
        y=apply_tone_kernel(self.x,self.p,color_filter=self.f)
        self.assertEqual(y.shape,self.x.shape)
        np.testing.assert_array_equal(self.x,source);np.testing.assert_array_equal(self.p.values,table)
        with self.assertRaises(ValueError):apply_tone_kernel_rgba([[1,2,3]],self.p)
        with self.assertRaises(ValueError):apply_tone_kernel_rgba([[.1,.2,.3,np.nan]],self.p)


if __name__=='__main__':unittest.main(verbosity=2)
