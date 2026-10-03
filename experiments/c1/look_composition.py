"""Float64 Look composition and deterministic single-camera-ICC variant plans.

Input and output are encoded sRGB channel values. The nominal input domain is
[0, 1], but finite extended values are accepted so a caller can explicitly choose
its own ICC-domain policy. Texture lookup clamps at texture edges; color-space
conversion and final output are neither clipped nor quantized. This module has
no RAW decoder, host API, native-ICC writer, gamut map or Delta-E attenuation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import numbers
from pathlib import Path
import re
from typing import Mapping

import numpy as np

from apps.local_looks.catalog import COLOR_FILTERS, LOOK_BINDINGS
from reproduction.ios_looks import color_spaces
from reproduction.ios_looks.renderer import TextureLUT, sample_texture

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESOURCE_DIR = ROOT / 'filters/looks'
MAIN_STRENGTHS = (25, 50, 75, 100)
_GROUP_FOLDERS = {'color': 'colors', 'monochrome': 'monochrome', 'artist': 'artist'}
_SAFE_ID = re.compile(r'[a-z][a-z0-9_]*\Z')


def _rgb(values):
    raw = np.asarray(values)
    if raw.ndim < 1 or raw.shape[-1] != 3 or raw.dtype.kind not in 'iuf':
        raise ValueError('Expected real numeric RGB with shape (..., 3)')
    with np.errstate(over='raise', invalid='raise'):
        try:
            rgb = raw.astype(np.float64, copy=False)
        except (FloatingPointError, OverflowError, ValueError) as error:
            raise ValueError('RGB must be representable as finite float64') from error
    if not np.isfinite(rgb).all():
        raise ValueError('RGB must be finite')
    return rgb


def _strength(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Real):
        raise ValueError('Strength must be a real scalar within 0..100')
    value = float(value)
    if not np.isfinite(value) or not 0 <= value <= 100:
        raise ValueError('Strength must be a finite scalar within 0..100')
    return value


def _convert(rgb, source, target):
    if source == target:
        return np.array(rgb, dtype=np.float64, copy=True)
    return color_spaces.convert_rgb(rgb, source, target, clip=False)


def compose_look(rgb, binding: Mapping, *, primary: TextureLUT | None = None,
                 secondary: TextureLUT | None = None,
                 prefilter: TextureLUT | None = None, strength=100) -> np.ndarray:
    """Evaluate one declared Look recipe with caller-provided texture tables.

    ``binding`` needs recipe/input_space/output_space. Single-table strength
    blends with the ORIGINAL input, after converting the Tone output to sRGB.
    Dual-table strength mixes secondary at zero to primary at one in the working
    domain. An optional prefilter is sampled before BOTH Tone textures. Resource
    precision restoration is performed by FloatLookCompositor when loading, not
    by this pure function. All returned arrays are independent float64 outputs.
    """
    source = _rgb(rgb)
    amount = _strength(strength)
    recipe = binding['recipe']
    input_space, output_space = binding['input_space'], binding['output_space']
    if input_space not in ('srgb', 'display-p3') or output_space not in ('srgb', 'display-p3'):
        raise ValueError('Look domains must be srgb or display-p3')
    if recipe not in ('identity', 'source_to_primary', 'secondary_to_primary'):
        raise ValueError('Unknown Look recipe')
    if recipe == 'identity':
        if primary is not None or secondary is not None or prefilter is not None:
            raise ValueError('Identity recipe cannot have texture tables')
        if amount != 100:
            raise ValueError('Original has no adjustable strength')
        return source.copy()
    if not isinstance(primary, TextureLUT):
        raise TypeError('A primary TextureLUT is required')
    if recipe == 'secondary_to_primary':
        if not isinstance(secondary, TextureLUT):
            raise TypeError('A dual recipe requires a secondary TextureLUT')
    elif secondary is not None:
        raise ValueError('A single-table recipe cannot have a secondary texture')
    if prefilter is not None and not isinstance(prefilter, TextureLUT):
        raise TypeError('Prefilter must be a TextureLUT')
    if recipe == 'source_to_primary' and amount == 0:
        return source.copy()
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            working = _convert(source, 'srgb', input_space)
            if prefilter is not None:
                working = sample_texture(prefilter, working)
            result = sample_texture(primary, working)
            weight = amount / 100
            if recipe == 'secondary_to_primary':
                low = sample_texture(secondary, working)
                result = low + (result - low) * weight
            result = _convert(result, output_space, 'srgb')
            if recipe == 'source_to_primary' and amount != 100:
                result = source + (result - source) * weight
    except FloatingPointError as error:
        raise ValueError('Look result exceeds finite float64 range') from error
    if not np.isfinite(result).all():
        raise ValueError('Look result must be finite')
    return np.array(result, dtype=np.float64, copy=True)


class FloatLookCompositor:
    """Lazy catalog-backed float compositor; table cache is instance-local.

    Hash checking is enabled for real catalog resources. Tests or callers using
    their own explicitly declared tables can disable it with verify_hashes=False.
    Grid sizes and precision policies are always validated. No image is opened.
    """
    def __init__(self, resource_dir=DEFAULT_RESOURCE_DIR, *, bindings=LOOK_BINDINGS,
                 color_filters=COLOR_FILTERS, verify_hashes=True):
        self.resource_dir = Path(resource_dir)
        self.bindings = bindings
        self.color_filters = color_filters
        self.verify_hashes = bool(verify_hashes)
        self._cache = {}

    def _table(self, spec):
        if spec is None:
            return None
        name = spec['filename']
        if (not isinstance(name, str) or not name.endswith('.cube') or name.startswith('.')
                or any(c in name for c in ('/', '\\', ':', '\x00'))):
            raise ValueError('Texture filename must be a plain .cube basename')
        grid = spec['grid_size']
        restore = spec['restore_half']
        if type(grid) is not int or grid < 2 or type(restore) is not bool:
            raise ValueError('Invalid texture grid or precision policy')
        digest = spec.get('sha256')
        key = (name, grid, restore, digest)
        if key not in self._cache:
            path = self.resource_dir / name
            if self.verify_hashes:
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                if actual != digest:
                    raise ValueError(f'Texture hash differs from catalog: {name}')
            table = TextureLUT.from_cube(path)
            if table.size != grid:
                raise ValueError(f'Texture grid differs from catalog: {name}')
            if restore:
                with np.errstate(over='raise', invalid='raise'):
                    try:
                        table = TextureLUT(table.values.astype(np.float16).astype(np.float64), table.name)
                    except FloatingPointError as error:
                        raise ValueError(f'Texture is not representable as binary16: {name}') from error
            self._cache[key] = table
        return self._cache[key]

    def apply(self, rgb, look_id: str, strength=100, color_filter: str | None = None) -> np.ndarray:
        """Compose a catalog Look into float64 sRGB without output quantization.

        Backend-declared attachments can be explicitly requested. The default
        ICC variant plan limits attachments to the six current UI mono Looks.
        """
        source = _rgb(rgb)
        amount = _strength(strength)
        if look_id not in self.bindings:
            raise ValueError(f'Unknown Look: {look_id}')
        binding = self.bindings[look_id]
        if color_filter is not None:
            if color_filter not in binding['filter_options'] or color_filter not in self.color_filters:
                raise ValueError(f'Attachment {color_filter!r} is unavailable for {look_id}')
        if binding['recipe'] == 'identity':
            return compose_look(source, binding, strength=amount)
        # Product zero is a true source bypass and needs no file access.
        if binding['recipe'] == 'source_to_primary' and amount == 0:
            return source.copy()
        return compose_look(source, binding,
                            primary=self._table(binding['primary_cube']),
                            secondary=self._table(binding['secondary_cube']),
                            prefilter=None if color_filter is None else self._table(self.color_filters[color_filter]['cube']),
                            strength=amount)

    def apply_variant(self, rgb, variant: 'LookVariant') -> np.ndarray:
        """Convenience for a native-ICC writer iterating build_variant_plan()."""
        return self.apply(rgb, variant.look_id, variant.strength, variant.color_filter)


@dataclass(frozen=True)
class LookVariant:
    """One independent camera-profile target; paths are relative POSIX/ASCII."""
    look_id: str
    strength: int
    color_filter: str | None
    group: str
    family: str
    relative_path: str
    description: str

    def to_dict(self):
        return asdict(self)


def _identifier(value):
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValueError('Variant identifiers must be lowercase ASCII identifiers')
    return ''.join(word.title() for word in value.split('_'))


def _variant(binding, strength, color_filter=None):
    look_id = binding['id']
    label = _identifier(look_id)
    group = binding['group']
    if group not in _GROUP_FOLDERS:
        raise ValueError('Unsupported profile group')
    suffix = '' if color_filter is None else '-Filter' + _identifier(color_filter)
    stem = f'LeicaQTyp116-LocalLooks-{label}-S{strength:03d}{suffix}-v1'
    folder = _GROUP_FOLDERS[group] if color_filter is None else f'attachments/{look_id}'
    dual = binding['recipe'] == 'secondary_to_primary'
    family = ('dual' if dual else 'single') + ('-p3' if binding['input_space'] == 'display-p3' else '-srgb')
    if color_filter is not None:
        family += '-prefilter'
    description = f'LocalLooks-Q1-{label}-S{strength:03d}{suffix}-v1'
    return LookVariant(look_id, strength, color_filter, group, family,
                       f'{folder}/{stem}.icm', description)


def build_variant_plan(bindings=LOOK_BINDINGS, color_filters=COLOR_FILTERS) -> tuple[LookVariant, ...]:
    """Return 206 default targets: 84 regular + 2 dual zero + 120 attachments.

    Original is excluded. Each of the 21 main Looks gets 25/50/75/100; dual
    endpoint Looks additionally get their meaningful zero endpoint. The six
    monochrome-group Looks get five declared attachments at those four strengths.
    Greg's backend attachments are intentionally absent from the UI-aligned plan.
    Iteration order follows the catalog and then each declared filter order.
    """
    plan = []
    for look_id, binding in bindings.items():
        if binding['id'] != look_id:
            raise ValueError('Look mapping key differs from catalog id')
        if binding['recipe'] == 'identity':
            if look_id != 'original':
                raise ValueError('Only Original may have identity recipe')
            continue
        if binding['recipe'] not in ('source_to_primary', 'secondary_to_primary'):
            raise ValueError('Unknown recipe in variant plan')
        strengths = (0, *MAIN_STRENGTHS) if binding['recipe'] == 'secondary_to_primary' else MAIN_STRENGTHS
        plan.extend(_variant(binding, s) for s in strengths)
        if binding['group'] == 'monochrome':
            for filter_id in binding['filter_options']:
                if filter_id not in color_filters:
                    raise ValueError('Unknown attachment in variant plan')
                plan.extend(_variant(binding, s, filter_id) for s in MAIN_STRENGTHS)
    paths = [v.relative_path for v in plan]
    descriptions = [v.description for v in plan]
    if (len(paths) != len({p.casefold() for p in paths})
            or len(descriptions) != len({d.casefold() for d in descriptions})):
        raise ValueError('Variant filenames or ICC descriptions collide')
    return tuple(plan)
