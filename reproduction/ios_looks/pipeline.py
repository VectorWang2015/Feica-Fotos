"""Explicit color-domain hypothesis runner around recovered Tone mathematics.

This API does NOT choose the FOTOS host pipeline. Every domain must be declared
by the experimenter; filenames and Look names are not treated as bindings.
"""
import numpy as np
if __package__:
    from .renderer import apply_tone_kernel
    from .color_spaces import convert_rgb, out_of_gamut
else:
    from renderer import apply_tone_kernel
    from color_spaces import convert_rgb, out_of_gamut


def _ranges(rgb):
    flat=np.asarray(rgb).reshape(-1,3)
    if not len(flat):
        return {'pixels':0,'min':None,'max':None,'outside_unit_rgb_pixels':0}
    return {'pixels':len(flat),'min':flat.min(axis=0).tolist(),'max':flat.max(axis=0).tolist(),
            'outside_unit_rgb_pixels':int(np.count_nonzero(out_of_gamut(flat,tolerance=1e-12)))}


def evaluate_hypothesis(rgb, primary, *, input_space, lut_space, output_space,
                        secondary=None, blend_factor=None, color_filter=None,
                        clip_output=False):
    """Return (float64 RGB, audit) with explicit unproven host-domain choices.

    convert(input -> LUT domain) -> optional filter -> single/double tone ->
    convert(LUT domain -> output) -> explicit optional final RGB-cube clipping.
    Texture address clamp remains intrinsic to each LUT sample; no extra
    pre-conversion clipping or gamut mapping is silently introduced.
    """
    if not isinstance(clip_output,(bool,np.bool_)):
        raise ValueError('clip_output must be an explicit boolean')
    # convert_rgb validates input shape/modes/finiteness and returns a copy.
    stage_input=convert_rgb(rgb,input_space,lut_space,clip=False)
    stage_output=apply_tone_kernel(stage_input,primary,secondary=secondary,
                                   blend_factor=blend_factor,color_filter=color_filter)
    converted=convert_rgb(stage_output,lut_space,output_space,clip=False)
    result=np.clip(converted,0,1) if clip_output else converted
    audit={
        'kind':'explicit-domain hypothesis, not recovered FOTOS host binding',
        'input_space':input_space,'lut_space':lut_space,'output_space':output_space,
        'topology':('filtered-' if color_filter is not None else '')+('double-tone' if secondary is not None else 'single-tone'),
        'primary_table':primary.name,'secondary_table':secondary.name if secondary is not None else None,
        'color_filter_table':color_filter.name if color_filter is not None else None,
        'blend_factor':float(blend_factor) if blend_factor is not None else None,
        'texture_sampling':'normalized linear clamp-to-edge, q=x*N-.5',
        'clip_output':bool(clip_output),
        'ranges':{'at_lut_input':_ranges(stage_input),'at_lut_output':_ranges(stage_output),
                  'output_before_clip':_ranges(converted),'output_after_clip':_ranges(result)},
        'not_claimed':['Actual App primary/secondary/colorFilter binding','Actual host color conversion location',
                       'RAW/preview input selection','Default spatial effects','GPU bit-exact arithmetic','App saving/encoder'],
    }
    return result,audit
