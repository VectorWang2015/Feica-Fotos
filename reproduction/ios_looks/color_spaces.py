"""Independent float64 standard RGB color-space mathematics (NumPy only).

This is a tool for comparing candidate spaces, NOT a recovered Leica FOTOS
CPU/host color pipeline or proof of how FOTOS reads Display P3 files. App
texture formats/bindings remain unproven/protected; extended GPU support is
NOT asserted. No ICC handling, file decoding, gamut mapping, LUTs or alpha.

Explicit modes (no default source/target and no alias guessing):
    srgb, display-p3                  -- encoded, shared sRGB transfer
    srgb-linear, display-p3-linear    -- linear light
Both spaces use D65 (x=0.3127, y=0.3290), but different RGB primaries.
Display P3 does NOT use the DCI-P3 gamma-2.6 transfer/white point.

All RGB functions accept finite real numeric arrays with shape (..., 3),
including a single (3,) triplet or an empty (..., 0, 3) batch. Values are
normalized channel values, not implicitly scaled byte codes. Results are
new float64 arrays of the same shape. Alpha/RGBA and complex/string/object/
bool arrays are rejected. Inputs are never modified.

Transfer functions reflect the standard positive curve about the origin
for negative values, as in W3C CSS Color 4 sample code. This mathematical
extension preserves out-of-gamut values, without claiming device support.
The published rounded breakpoints (0.04045 and 0.0031308) have a tiny
mismatch: encoded round trips close to the join can differ by about 3e-8.
No silent clipping: convert_rgb(..., clip=True) explicitly clips only the
final target RGB to [0, 1]. Overflow/non-finite results raise ValueError,
even with clip=True; extremely small values may underflow to signed zero.

Sources (retrieved 2026-10-02):
https://www.w3.org/TR/css-color-4/#color-conversion-code
https://www.w3.org/TR/css-color-4/#predefined-display-p3
https://developer.apple.com/documentation/coregraphics/cgcolorspace/displayp3
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "COLOR_SPACES", "convert_rgb", "srgb_eotf", "srgb_oetf",
    "linear_rgb_to_xyz_matrix", "linear_rgb_matrix", "out_of_gamut",
]

COLOR_SPACES = ("srgb", "display-p3", "srgb-linear", "display-p3-linear")
_MODES = {
    "srgb": ("srgb", True),
    "display-p3": ("display-p3", True),
    "srgb-linear": ("srgb", False),
    "display-p3-linear": ("display-p3", False),
}
_D65_XY = (0.3127, 0.3290)
_PRIMARIES_XY = {
    "srgb": ((0.640, 0.330), (0.300, 0.600), (0.150, 0.060)),
    "display-p3": ((0.680, 0.320), (0.265, 0.690), (0.150, 0.060)),
}


def _derive_rgb_to_xyz(primaries_xy):
    """Derive M from xy primaries, scaling columns to reproduce D65 Y=1."""
    xy = np.asarray(primaries_xy, dtype=np.float64)
    x, y = xy[:, 0], xy[:, 1]
    columns = np.stack((x / y, np.ones(3), (1.0 - x - y) / y))
    wx, wy = _D65_XY
    white = np.array((wx / wy, 1.0, (1.0 - wx - wy) / wy))
    scale = np.linalg.solve(columns, white)
    matrix = columns * scale
    matrix.setflags(write=False)
    return matrix


_RGB_TO_XYZ = {name: _derive_rgb_to_xyz(xy) for name, xy in _PRIMARIES_XY.items()}


def _mode(space):
    if not isinstance(space, str) or space not in _MODES:
        raise ValueError(f"color space must be one of {COLOR_SPACES}; got {space!r}")
    return _MODES[space]


def _linear_mode(space):
    basis, encoded = _mode(space)
    if encoded:
        raise ValueError("a linear RGB matrix requires an explicit '*-linear' mode")
    return basis


def _rgb(values):
    try:
        raw = np.asarray(values)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("RGB must be a regular real numeric array with shape (..., 3)") from exc
    if raw.ndim < 1 or raw.shape[-1] != 3:
        raise ValueError("RGB must have shape (..., 3); alpha is not part of the contract")
    if raw.dtype.kind not in "iuf":
        raise ValueError("RGB must contain real numeric values, not bool/complex/string/object")
    try:
        with np.errstate(over="raise", invalid="raise", under="ignore"):
            result = raw.astype(np.float64, copy=False)
    except (FloatingPointError, OverflowError, ValueError) as exc:
        raise ValueError("RGB values must be representable as finite float64") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError("RGB values must be finite")
    return result


def _transfer(rgb, *, decode):
    """Mask branches rather than evaluating unused powers (np.where)."""
    magnitude = np.abs(rgb)
    threshold = 0.04045 if decode else 0.0031308
    low = magnitude <= threshold
    result = np.empty_like(rgb)
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
            if decode:
                result[low] = rgb[low] / 12.92
                result[~low] = np.copysign(
                    ((magnitude[~low] + 0.055) / 1.055) ** 2.4, rgb[~low]
                )
            else:
                result[low] = 12.92 * rgb[low]
                result[~low] = np.copysign(
                    1.055 * magnitude[~low] ** (1.0 / 2.4) - 0.055, rgb[~low]
                )
    except FloatingPointError as exc:
        raise ValueError("transfer result exceeds finite float64 range") from exc
    if not np.all(np.isfinite(result)):
        raise ValueError("transfer result exceeds finite float64 range")
    return result


def srgb_eotf(rgb):
    """Decode encoded sRGB OR Display P3 (..., 3) values to linear light.

    abs(c) <= 0.04045: c / 12.92; otherwise sign(c)*((abs(c)+.055)/1.055)**2.4.
    No clipping, including negative/extended values.
    """
    return _transfer(_rgb(rgb), decode=True)


def srgb_oetf(rgb):
    """Encode linear sRGB OR Display P3 (..., 3), with reflected negatives.

    abs(c) <= .0031308: 12.92*c; otherwise sign(c)*(1.055*abs(c)**(1/2.4)-.055).
    No clipping, including negative/extended values.
    """
    return _transfer(_rgb(rgb), decode=False)


def linear_rgb_to_xyz_matrix(space):
    """Return a fresh 3x3 linear RGB -> XYZ(D65, Ywhite=1) matrix.

    Only 'srgb-linear' and 'display-p3-linear' are accepted. Matrix convention
    is column-vector XYZ = M @ RGB; row/batched arrays use rgb @ M.T.
    Coefficients are derived from xy chromaticities, not fitted to App data.
    """
    return _RGB_TO_XYZ[_linear_mode(space)].copy()


def linear_rgb_matrix(source, target):
    """Return a fresh 3x3 source-linear -> target-linear matrix (M @ RGB).

    Explicit linear modes only. Both whites are D65, so no chromatic
    adaptation is needed. This matrix MUST NOT be applied to encoded RGB.
    """
    src, dst = _linear_mode(source), _linear_mode(target)
    if src == dst:
        return np.eye(3, dtype=np.float64)
    return np.linalg.solve(_RGB_TO_XYZ[dst], _RGB_TO_XYZ[src])


def convert_rgb(rgb, source, target, *, clip=False):
    """Convert explicit encoded/linear sRGB/Display P3 modes.

    Order: decode if needed -> linear primaries matrix if needed -> encode
    if needed -> optional final clipping. No gamut mapping or alpha.
    Default output retains out-of-gamut numbers; use out_of_gamut() to
    inspect them. clip must be an explicit bool (not a truthy string).
    Same-mode conversions return a new array without a transfer round trip.
    Non-finite input, malformed shape/mode or float64 overflow: ValueError.
    """
    src, src_encoded = _mode(source)
    dst, dst_encoded = _mode(target)
    if not isinstance(clip, (bool, np.bool_)):
        raise ValueError("clip must be a boolean")
    values = _rgb(rgb)
    if source == target:
        result = values.copy()
    else:
        result = _transfer(values, decode=True) if src_encoded else values.copy()
        if src != dst:
            matrix = linear_rgb_matrix(src + "-linear", dst + "-linear")
            try:
                with np.errstate(over="raise", invalid="raise", divide="raise", under="ignore"):
                    result = result @ matrix.T
            except FloatingPointError as exc:
                raise ValueError("matrix result exceeds finite float64 range") from exc
            if not np.all(np.isfinite(result)):
                raise ValueError("matrix result exceeds finite float64 range")
        if dst_encoded:
            result = _transfer(result, decode=False)
    return np.clip(result, 0.0, 1.0) if clip else result


def out_of_gamut(rgb, *, tolerance=0.0):
    """Per-triplet mask for any channel outside [-tolerance, 1+tolerance].

    Shape is (...) (scalar np.bool_ for one triplet). This tests the numeric
    RGB cube in the already-declared space, not a device profile/gamut.
    Default is exact; e.g. tolerance=1e-12 may ignore matrix roundoff at white.
    No values are clipped or changed.
    """
    if (isinstance(tolerance, (bool, np.bool_))
            or not isinstance(tolerance, (int, float, np.integer, np.floating))):
        raise ValueError("tolerance must be a finite nonnegative scalar")
    try:
        tolerance = float(tolerance)
    except (OverflowError, ValueError) as exc:
        raise ValueError("tolerance must be a finite nonnegative scalar") from exc
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be a finite nonnegative scalar")
    values = _rgb(rgb)
    return np.any((values < -tolerance) | (values > 1.0 + tolerance), axis=-1)
