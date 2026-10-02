#!/usr/bin/env python3
"""Independent CPU reference for the recovered iOS RGB LUT stage.

Not a RAW decoder or full FOTOS emulator. Input is explicitly encoded-sRGB RGB.
Standard CUBE resources retain their samples; evaluation follows normalized
3D texture coordinates q = x*N - 0.5, trilinear filtering, clamp-to-edge.
Steve mixes endpoint outputs, not identity and a single table.
Vivid100 alone restores recovered CUBE decimal samples to source binary16
values before float64 evaluation; no unvalidated Vivid strength rule.
"""
from dataclasses import dataclass
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESOURCE_DIR = ROOT / 'filters/looks'
STEVE1 = '29_Leica-Looks_SteveMcCurry_sRGB_sRGB_TESTFLIGHT-LUT-1.cube'
STEVE3 = '29_Leica-Looks_SteveMcCurry_sRGB_sRGB_TESTFLIGHT-LUT-3.cube'
ETERNAL = 'Leica-Looks_Eternal_sRGB_sRGB_Release.cube'
VIVID = 'Leica_Vivid_sRGB_sRGB_Release.cube'


@dataclass(frozen=True)
class TextureLUT:
    """Table indexing is [blue, green, red, output-channel], R varies fastest."""
    values: np.ndarray
    name: str = ''

    def __post_init__(self):
        a = np.asarray(self.values, dtype=np.float64)
        if a.ndim != 4 or a.shape[-1] != 3 or len(set(a.shape[:3])) != 1 or a.shape[0] < 2:
            raise ValueError('Expected cubic [B,G,R,3] table of at least2 nodes')
        if not np.isfinite(a).all():
            raise ValueError('Non-finite table value')
        a = a.copy(); a.setflags(write=False)
        object.__setattr__(self, 'values', a)

    @property
    def size(self):
        return self.values.shape[0]

    @classmethod
    def from_cube(cls, path):
        path = Path(path); n = None; rows = []; domain_lo = [0., 0., 0.]; domain_hi = [1., 1., 1.]
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            line = line.split('#', 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if parts[0] == 'TITLE':
                continue
            if parts[0] == 'LUT_3D_SIZE':
                if n is not None:
                    raise ValueError('Duplicate size declaration')
                n = int(parts[1]); continue
            if parts[0] == 'DOMAIN_MIN':
                domain_lo = list(map(float, parts[1:])); continue
            if parts[0] == 'DOMAIN_MAX':
                domain_hi = list(map(float, parts[1:])); continue
            if len(parts) != 3:
                raise ValueError('Unexpected record')
            rows.append([float(x) for x in parts])
        if domain_lo != [0., 0., 0.] or domain_hi != [1., 1., 1.]:
            raise ValueError('Only normalized0..1 domain supported by this reference')
        if n is None or len(rows) != n**3:
            raise ValueError('Invalid sample count')
        return cls(np.asarray(rows).reshape(n, n, n, 3), path.name)


def sample_texture(table, rgb):
    """Pure trilinear normalized sampling. No gamma conversion or global curve."""
    x = np.asarray(rgb, dtype=np.float64)
    if x.ndim < 1 or x.shape[-1] != 3 or not np.isfinite(x).all():
        raise ValueError('Expected finite [...,3] RGB array')
    n = table.size
    q = np.clip(np.clip(x, 0, 1)*n - .5, 0, n-1)
    lower = np.floor(q).astype(np.int32)
    upper = np.minimum(lower+1, n-1)
    f = q-lower
    r0, g0, b0 = np.moveaxis(lower, -1, 0)
    r1, g1, b1 = np.moveaxis(upper, -1, 0)
    fr, fg, fb = np.moveaxis(f, -1, 0)
    def mix(a, b, weight):
        return a+(b-a)*weight[..., None]
    v = table.values
    c00 = mix(v[b0,g0,r0], v[b0,g0,r1], fr)
    c01 = mix(v[b0,g1,r0], v[b0,g1,r1], fr)
    c10 = mix(v[b1,g0,r0], v[b1,g0,r1], fr)
    c11 = mix(v[b1,g1,r0], v[b1,g1,r1], fr)
    return mix(mix(c00,c01,fg), mix(c10,c11,fg), fb)


def mix_tables(secondary, primary, rgb, factor):
    if not np.isfinite(factor) or not 0 <= factor <= 1:
        raise ValueError('Factor must be within0..1')
    a = sample_texture(secondary, rgb)
    b = sample_texture(primary, rgb)
    return a+(b-a)*factor


def apply_tone_kernel(rgb, primary, *, secondary=None, blend_factor=None, color_filter=None):
    """Reference for the four recovered Tone kernels; no inferred Look binding.

    Optional color_filter runs BEFORE either tone table. The generic shader
    factor may be any finite scalar: the recovered kernel does not clamp it.
    The higher-level user-strength wrapper separately limits its0..100 range.
    Input/output numbers are in the caller-declared texture working domain.
    This function performs no gamma/primary conversion or output clamp.
    """
    if not isinstance(primary, TextureLUT):
        raise TypeError('primary must be a TextureLUT')
    if secondary is not None and not isinstance(secondary, TextureLUT):
        raise TypeError('secondary must be a TextureLUT')
    if color_filter is not None and not isinstance(color_filter, TextureLUT):
        raise TypeError('color_filter must be a TextureLUT')
    if secondary is None and blend_factor is not None:
        raise ValueError('A blend factor requires both tone tables')
    if secondary is not None:
        if blend_factor is None or np.ndim(blend_factor) != 0 or not np.isfinite(blend_factor):
            raise ValueError('Two-table kernel requires an explicit finite scalar factor')
        factor = float(blend_factor)
    coordinates = sample_texture(color_filter, rgb) if color_filter is not None else rgb
    a = sample_texture(primary, coordinates)
    if secondary is None:
        return a
    b = sample_texture(secondary, coordinates)
    return b+(a-b)*factor


def apply_tone_kernel_rgba(rgba, primary, *, secondary=None, blend_factor=None, color_filter=None):
    """RGBA façade: recovered Tone kernels use source RGB and write alpha1.

    No automatic premultiply/unpremultiply or source-alpha compositing.
    """
    x = np.asarray(rgba, dtype=np.float64)
    if x.ndim < 1 or x.shape[-1] != 4 or not np.isfinite(x).all():
        raise ValueError('Expected finite [...,4] RGBA array')
    rgb = apply_tone_kernel(x[..., :3], primary, secondary=secondary,
                           blend_factor=blend_factor, color_filter=color_filter)
    return np.concatenate([rgb, np.ones((*rgb.shape[:-1], 1), dtype=np.float64)], axis=-1)


def quantize_u8(rgb):
    return np.floor(np.clip(rgb, 0, 1)*255+.5).astype(np.uint8)


def render_rgb8(rgb, look, strength=100, resource_dir=RESOURCE_DIR, chunk_pixels=200_000):
    x = np.asarray(rgb)
    if x.dtype != np.uint8 or x.ndim != 3 or x.shape[-1] != 3:
        raise ValueError('This wrapper accepts H×W×3 uint8 encodedRGB only')
    if chunk_pixels < 1:
        raise ValueError('Invalid chunk size')
    resources = Path(resource_dir)
    if look == 'steve':
        secondary = TextureLUT.from_cube(resources/STEVE1)
        primary = TextureLUT.from_cube(resources/STEVE3)
        if not 0 <= strength <= 100:
            raise ValueError('Steve strength must be0..100')
    elif look == 'eternal':
        if strength != 100:
            raise ValueError('Only Eternal100 is validated; no invented intensity rule')
        primary = TextureLUT.from_cube(resources/ETERNAL)
    elif look == 'vivid':
        if strength != 100:
            raise ValueError('Only Vivid100 is validated; no invented intensity rule')
        recovered = TextureLUT.from_cube(resources/VIVID)
        # Frozen Vivid contract: recover exact source half samples, then retain
        # existing float64 sampler/quantizer. Do not apply this to other Looks.
        primary = TextureLUT(recovered.values.astype(np.float16).astype(np.float64), recovered.name)
    else:
        raise ValueError('Unknown supported look')
    flat = x.reshape(-1,3); output = np.empty_like(flat)
    for start in range(0,len(flat),chunk_pixels):
        xx = flat[start:start+chunk_pixels].astype(np.float64)/255
        yy = mix_tables(secondary,primary,xx,strength/100) if look == 'steve' else sample_texture(primary,xx)
        output[start:start+len(xx)] = quantize_u8(yy)
    return output.reshape(x.shape)


def inspect_input_color_tags(info, exif_color_space=None):
    """Describe tags without transforming pixels; return explicit contract conflicts."""
    icc = info.get('icc_profile')
    tags = {'icc_present': bool(icc),
            'icc_sha256': hashlib.sha256(icc).hexdigest() if icc else None,
            'png_gamma': info.get('gamma'),
            'png_srgb_rendering_intent': info.get('srgb'),
            'png_chromaticity': list(info['chromaticity']) if 'chromaticity' in info else None,
            'exif_color_space': exif_color_space}
    conflicts = []
    if icc:
        conflicts.append('Embedded ICC is not interpreted or converted by this RGB-stage reference')
    if tags['png_gamma'] is not None and abs(float(tags['png_gamma'])-.45455) > .0002:
        conflicts.append('PNG gamma is inconsistent with the asserted encoded-sRGB input')
    if tags['png_chromaticity'] is not None:
        standard = np.asarray([.3127,.329,.64,.33,.30,.60,.15,.06])
        value = np.asarray(tags['png_chromaticity'])
        if value.shape != (8,) or not np.allclose(value,standard,atol=.0002,rtol=0):
            conflicts.append('PNG chromaticities are not the standard sRGB primaries/white')
    if exif_color_space not in (None,1):
        conflicts.append('EXIF color space is not explicitly sRGB')
    return tags, conflicts


def main():
    from PIL import Image, PngImagePlugin
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--look', choices=['steve','eternal','vivid'], required=True)
    parser.add_argument('--strength', type=float, default=100)
    parser.add_argument('--input-domain', choices=['encoded-srgb'], required=True,
                        help='Explicit input contract; no implicit ICC/RAW conversion')
    parser.add_argument('--resource-dir', type=Path, default=RESOURCE_DIR)
    parser.add_argument('--acknowledge-unconverted-color-tags', action='store_true',
                        help='Explicitly assert encoded-sRGB numbers despite conflicting/uninterpreted input tags; does not color-convert')
    args = parser.parse_args()
    if args.output.suffix.lower() != '.png':
        parser.error('Reference output is losslessRGB8 PNG only')
    sidecar = Path(str(args.output)+'.json')
    if args.output.exists() or sidecar.exists():
        parser.error('Output or manifest already exists; never overwrite')
    raw = args.input.read_bytes()
    if raw.startswith(b'\x89PNG\r\n\x1a\n'):
        if len(raw) < 26 or raw[24] != 8:
            parser.error('Only8-bit PNG input supported; do not silently truncate16-bit')
    elif not raw.startswith(b'\xff\xd8'):
        parser.error('OnlyJPEG or8-bit PNG accepted; not DNG/RAW')
    with Image.open(args.input) as im:
        if im.mode != 'RGB':
            parser.error('RequireRGB image; no silent palette/CMYK/alpha conversion')
        orientation = im.getexif().get(274,1)
        if orientation != 1:
            parser.error('Supply an upright image with orientation1/absent; orientation is a separate stage')
        try:
            exif_color_space = im.getexif().get_ifd(34665).get(40961)
        except (KeyError, TypeError, ValueError):
            exif_color_space = None
        input_color_tags, conflicts = inspect_input_color_tags(im.info, exif_color_space)
        if conflicts and not args.acknowledge_unconverted_color_tags:
            parser.error('; '.join(conflicts)+'; color-convert externally or explicitly acknowledge the numeric-domain assertion')
        source = np.asarray(im)
    rendered = render_rgb8(source,args.look,args.strength,args.resource_dir)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    metadata = PngImagePlugin.PngInfo(); metadata.add(b'sRGB',bytes([0]))
    metadata.add_text('Description','Local reference LUT stage; not a complete FOTOS/RAW implementation')
    with args.output.open('xb') as f:
        Image.fromarray(rendered).save(f,format='PNG',pnginfo=metadata)
    names = {'steve': [STEVE1, STEVE3], 'eternal': [ETERNAL], 'vivid': [VIVID]}[args.look]
    manifest = {'input':str(args.input.resolve()),'input_sha256':hashlib.sha256(raw).hexdigest(),
                'output':str(args.output.resolve()),'output_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),
                'look':args.look,'strength':args.strength,'input_domain':args.input_domain,
                'input_color_tags':input_color_tags,'unconverted_tag_warnings':conflicts,
                'tag_override_acknowledged':args.acknowledge_unconverted_color_tags,
                'output_domain':'encoded-sRGB, RGB8 PNG with sRGB chunk',
                'quantization':'clip0..1, floor(value*255+0.5), no dithering',
                'sampler':'normalized coordinate x*N-.5; trilinear; clamp-to-edge',
                'table_values':('CUBE decimal samples restored via float16 then float64; frozen Vivid full-weight1 contract'
                                if args.look == 'vivid' else 'CUBE decimal samples retained as float64'),
                'resource_hashes':{n:hashlib.sha256((args.resource_dir/n).read_bytes()).hexdigest() for n in names},
                'not_included':['RAW/image source selection','automatic ICC conversion','grain/spatial effects','Photos saving behavior','Apple JPEG encoder'],
                'validation_scope':'Steve0/25/50/75/100 numerically corroborated across two Q1 scenes (user labels; 0/25/75 sequence additionally ranked); Eternal100 corroborated; Vivid100 frozen full-weight binary16 model corroborated on new0333 user-labeled scene without fitting; other Vivid strengths unsupported; remaining continuous/UI bindings not CPU-traced'}
    with sidecar.open('x') as f:json.dump(manifest,f,indent=2)
    print(json.dumps({'output':str(args.output),'manifest':str(sidecar),'width':rendered.shape[1],'height':rendered.shape[0]},indent=2))

if __name__ == '__main__':
    main()
