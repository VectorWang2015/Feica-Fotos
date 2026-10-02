"""Offline RGB8 backend for Local Looks (not an official FOTOS implementation).

The existing LeicaRGB full-weight reference stage and Steve endpoint mix are
reused unchanged. Additional catalog entries, P3 working spaces, attachments
and source-blend strengths are app candidate recipes, not a claim about official
FOTOS behavior. No RAW development, Qt, network access, bundled vendor tables,
research sidecars, or in-place writes.
``resource_dir`` is mandatory; only the selected Look's tables are read lazily.
Progress callbacks run synchronously on the caller's thread. Cancellation uses
``threading.Event`` and is checked around every <=200,000-pixel render chunk.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import copy
import io
import math
import numbers
import os
import stat
import struct
import threading
import warnings

import numpy as np
from PIL import Image, ImageCms, ImageFile, PngImagePlugin

from reproduction.ios_looks import color_spaces, dng_preview, renderer
from .catalog import (
    LookSpec, LOOKS, LOOK_BY_ID, LOOK_BINDINGS, LOOK_GROUPS, GROUP_TITLES,
    LOOK_PREVIEW, LOOK_STRENGTH_RULES, LOOK_FILTER_OPTIONS, COLOR_FILTERS, look_preview,
)

MAX_PIXELS = 64_000_000
MAX_FILE_BYTES = 512 * 1024 * 1024
PREVIEW_EDGE = 1600
CHUNK_PIXELS = 200_000
SOFTWARE = "Local Looks — independent RGB reference"
DNG_SELECTION_POLICY = (
    "Local Looks largest supported preview; not claimed FOTOS universal policy"
)


class EngineError(Exception):
    """Base class for user-facing backend failures."""


class ResourceError(EngineError):
    """A required, explicitly configured LUT is missing or invalid."""


class ImageLoadError(EngineError, ValueError):
    """An input cannot safely be interpreted as supported upright RGB8."""


class ExportError(EngineError, OSError):
    """A new image could not be safely exported; never authorizes overwrite."""


class CancelledError(EngineError):
    """The caller cancelled an operation; this is not an image/resource error."""


Progress = Callable[[float], None]
# On Python <3.14 warnings.catch_warnings is process-global. Serialize our load
# scopes (including the existing DNG adapter's scope), not render/export tasks.
_IMAGE_LOAD_LOCK = threading.RLock()


def _rgb8(value: np.ndarray) -> np.ndarray:
    rgb = np.asarray(value)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("Expected H×W×3 uint8 RGB8; no implicit bit-depth conversion")
    if not rgb.shape[0] or not rgb.shape[1] or rgb.shape[0] * rgb.shape[1] > MAX_PIXELS:
        raise ValueError("Image dimensions must be positive and at most 64 MP")
    return rgb


def _immutable_rgb(value: np.ndarray) -> np.ndarray:
    rgb = _rgb8(value)
    # A bytes backing store is genuinely immutable, unlike setflags(False) on
    # an owning ndarray (whose writeability the caller could simply re-enable).
    return np.frombuffer(rgb.tobytes(order="C"), dtype=np.uint8).reshape(rgb.shape)


@dataclass(frozen=True)
class ImageDocument:
    path: Path
    rgb: np.ndarray
    preview_rgb: np.ndarray
    source_kind: str
    source_info: dict
    width: int
    height: int

    def __post_init__(self):
        rgb, preview = _rgb8(self.rgb), _rgb8(self.preview_rgb)
        if (self.width, self.height) != (rgb.shape[1], rgb.shape[0]):
            raise ValueError("Document dimensions disagree with RGB pixels")
        if max(preview.shape[:2]) > PREVIEW_EDGE:
            raise ValueError("Document preview must have a maximum edge of 1600")
        object.__setattr__(self, "path", Path(self.path).expanduser().resolve())
        object.__setattr__(self, "rgb", _immutable_rgb(rgb))
        object.__setattr__(self, "preview_rgb", _immutable_rgb(preview))
        object.__setattr__(self, "source_info", copy.deepcopy(dict(self.source_info)))


def _check_cancel(cancel: threading.Event | None) -> None:
    if cancel is not None and cancel.is_set():
        raise CancelledError("Operation cancelled")


def _report(progress: Progress | None, value: float, cancel: threading.Event | None) -> None:
    _check_cancel(cancel)
    if progress is not None:
        progress(float(value))
    _check_cancel(cancel)


def _look_strength(look_id: str, strength: float | None) -> tuple[LookSpec, float]:
    try:
        look = LOOK_BY_ID[look_id]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Unknown Look: {look_id!r}") from error
    if strength is None:
        strength = look.default_strength
    if isinstance(strength, (bool, np.bool_)) or not isinstance(strength, numbers.Real):
        raise ValueError("Strength must be a finite number within 0..100")
    strength = float(strength)
    if not math.isfinite(strength) or not 0 <= strength <= 100:
        raise ValueError("Strength must be a finite number within 0..100")
    if not look.adjustable and strength != look.default_strength:
        raise ValueError(f"{look.title} only supports strength {look.default_strength}")
    return look, strength


def _read_image_bytes(path: Path) -> bytes:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(fd, "rb") as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ImageLoadError("Input must be a regular image file")
        if not 8 <= before.st_size <= MAX_FILE_BYTES:
            raise ImageLoadError("Input must be between 8 bytes and 512 MiB")
        raw = handle.read(MAX_FILE_BYTES + 1)
        after = os.fstat(handle.fileno())
    def stamp(s):
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns
    if len(raw) != before.st_size or stamp(before) != stamp(after):
        raise ImageLoadError("Input changed during read; reload the original")
    return raw


def _jpeg_details(raw: bytes) -> tuple[tuple[int, int], bytes | None]:
    """Bounded JPEG marker walk: reject non-8-bit and incomplete ICC profiles.

    Include ICC chunks after SOS, which Pillow's header-only metadata inventory
    otherwise misses. Entropy decoding remains Pillow's job. Gray/RGB/CMYK8
    frames are accepted here; unprofiled CMYK is rejected before conversion.
    """
    if not raw.startswith(b"\xff\xd8") or not raw.endswith(b"\xff\xd9"):
        raise ImageLoadError("JPEG must be a complete SOI/EOI stream; truncated image rejected")
    pos, frame, scans, marker_count = 2, None, 0, 0
    parts, declared_parts = {}, None
    sof_codes = {0xc0, 0xc1, 0xc2, 0xc3, 0xc5, 0xc6, 0xc7,
                 0xc9, 0xca, 0xcb, 0xcd, 0xce, 0xcf}
    while pos < len(raw):
        marker_count += 1
        if marker_count > 1_000_000 or raw[pos] != 0xff:
            raise ImageLoadError("Invalid or excessive JPEG marker sequence")
        while pos < len(raw) and raw[pos] == 0xff:
            pos += 1
        if pos == len(raw):
            break
        code = raw[pos]
        pos += 1
        if code == 0xd9:
            if pos != len(raw) or frame is None or not scans:
                raise ImageLoadError("Premature or trailing JPEG EOI")
            if declared_parts is not None and set(parts) != set(range(1, declared_parts + 1)):
                raise ImageLoadError("Incomplete JPEG ICC profile; cannot assume sRGB")
            return frame, b"".join(parts[n] for n in sorted(parts)) if parts else None
        if code in (0, 1, 0xd8) or 0xd0 <= code <= 0xd7 or pos + 2 > len(raw):
            raise ImageLoadError("Invalid JPEG marker")
        length = int.from_bytes(raw[pos:pos + 2], "big")
        if length < 2 or length > len(raw) - pos:
            raise ImageLoadError("Truncated JPEG segment")
        data = raw[pos + 2:pos + length]
        if code == 0xe2 and data.startswith(b"ICC_PROFILE\x00"):
            if len(data) <= 14:
                raise ImageLoadError("Empty or truncated JPEG ICC profile")
            sequence, total = data[12:14]
            if not 1 <= sequence <= total or sequence in parts or declared_parts not in (None, total):
                raise ImageLoadError("Invalid JPEG ICC chunk sequence")
            declared_parts, parts[sequence] = total, data[14:]
        if code in sof_codes:
            if frame is not None or len(data) < 6:
                raise ImageLoadError("Invalid JPEG frame")
            bits, height, width, components = (data[0], int.from_bytes(data[1:3], "big"),
                                               int.from_bytes(data[3:5], "big"), data[5])
            if bits != 8 or code not in (0xc0, 0xc2) or components not in (1, 3, 4):
                raise ImageLoadError("Only 8-bit baseline/progressive JPEG is supported")
            if len(data) != 6 + 3 * components or not width or not height or width * height > MAX_PIXELS:
                raise ImageLoadError("JPEG dimensions/component count invalid or exceed 64 MP")
            frame = width, height
        pos += length
        if code == 0xda:
            if frame is None:
                raise ImageLoadError("JPEG scan precedes frame")
            scans += 1
            while True:
                escape = raw.find(b"\xff", pos)
                if escape < 0:
                    raise ImageLoadError("Truncated JPEG entropy scan")
                cursor = escape + 1
                while cursor < len(raw) and raw[cursor] == 0xff:
                    cursor += 1
                if cursor == len(raw):
                    raise ImageLoadError("Truncated JPEG entropy marker")
                if raw[cursor] == 0 or 0xd0 <= raw[cursor] <= 0xd7:
                    pos = cursor + 1
                else:
                    pos = escape
                    break
    raise ImageLoadError("Missing JPEG EOI")


def _load_raster(path: Path) -> tuple[np.ndarray, str, dict]:
    raw = _read_image_bytes(path)
    is_png = path.suffix.lower() == ".png"
    expected_size, jpeg_icc = None, None
    if is_png:
        if len(raw) < 33 or raw[:8] != b"\x89PNG\r\n\x1a\n" or raw[12:16] != b"IHDR":
            raise ImageLoadError("Invalid PNG header")
        if raw[24] != 8 or raw[25] != 2:
            raise ImageLoadError("Only RGB PNG8 is supported; no 16-bit/alpha/palette truncation")
    else:
        expected_size, jpeg_icc = _jpeg_details(raw)
    if ImageFile.LOAD_TRUNCATED_IMAGES:
        raise ImageLoadError("Pillow truncated-image decoding must remain disabled")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        warnings.simplefilter("error", UserWarning)
        # verify() catches PNG integrity errors; both handles decode from the
        # same in-memory snapshot, not a second filesystem read.
        with Image.open(io.BytesIO(raw)) as check:
            if check.width * check.height > MAX_PIXELS:
                raise ImageLoadError("Image exceeds the 64 MP limit")
            check.verify()
        with Image.open(io.BytesIO(raw)) as image:
            if image.format != ("PNG" if is_png else "JPEG"):
                raise ImageLoadError("Image contents do not match the supported file extension")
            if expected_size is not None and image.size != expected_size:
                raise ImageLoadError("JPEG frame and decoded dimensions disagree")
            if image.mode not in (("RGB",) if is_png else ("RGB", "L", "CMYK")):
                raise ImageLoadError(f"Unsupported image mode {image.mode}; RGB8 required")
            if is_png and "transparency" in image.info:
                raise ImageLoadError("PNG transparency is unsupported; supply opaque RGB PNG8")
            try:
                exif = image.getexif()
                orientation = exif.get(274, 1)
                # Read only the standard EXIF IFD, not private MakerNote/GPS or
                # UserComment structures. An unreadable IFD is not evidence of
                # absent color metadata and must not silently imply sRGB.
                exif_ifd = exif.get_ifd(34665) if 34665 in exif else {}
                exif_color_space = exif_ifd.get(40961, exif.get(40961))
            except (TypeError, ValueError, KeyError, SyntaxError, struct.error, UserWarning) as error:
                raise ImageLoadError(f"Cannot parse EXIF orientation/color metadata: {error}") from error
            if type(orientation) is not int or orientation not in dng_preview.ORIENTATION:
                raise ImageLoadError("EXIF Orientation must be an integer in 1..8")
            icc = image.info.get("icc_profile") if is_png else jpeg_icc
            if is_png and "icc_profile" in image.info and not icc:
                raise ImageLoadError("Empty or invalid embedded ICC profile; cannot assume sRGB")
            image.load()
            info = {"stored_size": list(image.size), "input_mode": image.mode,
                    "orientation": orientation, "output_orientation": 1,
                    "orientation_action": dng_preview.ORIENTATION[orientation][1],
                    "icc_present": bool(icc), "exif_color_space": exif_color_space,
                    "input_domain": "encoded-srgb"}
            if icc:
                try:
                    source_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
                    image_rgb = ImageCms.profileToProfile(
                        image, source_profile, ImageCms.createProfile("sRGB"), outputMode="RGB")
                except (OSError, ValueError, TypeError, ImageCms.PyCMSError) as error:
                    raise ImageLoadError(f"Embedded ICC conversion to sRGB failed: {error}") from error
                info["color_conversion"] = "Embedded ICC to standard sRGB via Pillow ImageCms"
            else:
                if image.mode == "CMYK":
                    raise ImageLoadError("CMYK JPEG without an ICC profile cannot be interpreted as sRGB")
                _, conflicts = renderer.inspect_input_color_tags(image.info, exif_color_space)
                if conflicts:
                    raise ImageLoadError(
                        "Image color tags conflict with sRGB: " + "; ".join(conflicts)
                        + ". Supply a color-converted sRGB image or a valid embedded ICC profile.")
                image_rgb = image.convert("RGB") if image.mode == "L" else image
                info["color_conversion"] = "none" if image.mode == "RGB" else "grayscale expanded to RGB"
                info["assumption"] = "No embedded ICC; interpreted as encoded sRGB"
            transpose = dng_preview.ORIENTATION[orientation][0]
            if transpose is not None:
                image_rgb = image_rgb.transpose(transpose)
            return np.array(image_rgb, copy=True), "PNG" if is_png else "JPEG", info


class ImageEngine:
    """Thread-safe resource cache; each call owns its image/decoder state."""

    def __init__(self, resource_dir: str | os.PathLike):
        if resource_dir is None:
            raise ValueError("resource_dir must be configured explicitly")
        self.resource_dir = Path(resource_dir).expanduser().resolve()
        self._cache: dict[tuple[str, bool], renderer.TextureLUT] = {}
        self._cache_lock = threading.Lock()

    def _tables(self, look_id: str, color_filter: str | None = None) -> tuple:
        """Return (primary, secondary, optional filter), loading selected files only."""
        binding = LOOK_BINDINGS[look_id]
        if color_filter is not None and color_filter not in LOOK_FILTER_OPTIONS[look_id]:
            raise ValueError(f"Color filter {color_filter!r} is not allowed for {look_id}")
        cubes = (binding["primary_cube"], binding["secondary_cube"],
                 COLOR_FILTERS[color_filter]["cube"] if color_filter is not None else None)
        result = []
        with self._cache_lock:
            for cube in cubes:
                if cube is None:
                    result.append(None)
                    continue
                name = cube["filename"]
                # Include precision policy so one shared filename could not
                # accidentally reuse decimal/half samples under another recipe.
                key = (name, cube["restore_half"])
                if key not in self._cache:
                    path = self.resource_dir / name
                    try:
                        table = renderer.TextureLUT.from_cube(path)
                        if cube["restore_half"]:
                            # Vivid's frozen behavior is retained. Newly added
                            # candidates/attachments use explicit half recovery;
                            # Steve and Eternal remain untouched decimal tables.
                            table = renderer.TextureLUT(
                                table.values.astype(np.float16).astype(np.float64), table.name)
                    except (OSError, ValueError, OverflowError, IndexError) as error:
                        raise ResourceError(
                            f"Required {look_id} LUT missing or invalid: {path}. "
                            "Configure your own authorized local resources; no official renderer is bundled. "
                            f"({error})") from error
                    self._cache[key] = table
                result.append(self._cache[key])
        return tuple(result)

    def load_document(self, path: str | os.PathLike) -> ImageDocument:
        """Read an original without modifying it; return immutable upright RGB8.

        DNG uses the existing bounded adapter's inspect/extract calls. Because
        those public APIs each read the file, their source hashes must agree.
        Ties for the greatest eligible area are refused, never guessed.
        """
        with _IMAGE_LOAD_LOCK:
            return self._load_document(path)

    def _load_document(self, path: str | os.PathLike) -> ImageDocument:
        path = Path(path).expanduser().resolve()
        try:
            if path.suffix.lower() == ".dng":
                limits = dng_preview.Limits(max_file_bytes=MAX_FILE_BYTES, max_pixels=MAX_PIXELS)
                inventory = dng_preview.inspect_dng(path, limits=limits)
                eligible = [d for d in inventory["directories"] if d["eligible"]]
                if not eligible:
                    reasons = "; ".join(d.get("rejection_reason", "unsupported")
                                        for d in inventory["directories"][:4])
                    raise ImageLoadError("No supported DNG JPEG preview (no CFA/RAW development): " + reasons)
                area = lambda d: d["width"][0] * d["height"][0]
                largest = max(map(area, eligible))
                choices = [d for d in eligible if area(d) == largest]
                if len(choices) != 1:
                    offsets = ", ".join(str(d["ifd_offset"]) for d in choices)
                    raise ImageLoadError(
                        f"Ambiguous largest DNG preview (IFDs {offsets}); explicit selection required")
                rgb, info = dng_preview.extract_preview_rgb8(
                    path, choices[0]["ifd_offset"], input_domain="encoded-srgb",
                    acknowledge_unconverted_color_tags=False, limits=limits)
                if info["input_sha256"] != inventory["input_sha256"]:
                    raise ImageLoadError("DNG changed between inspection and extraction; reload the original")
                info["selection_policy"] = DNG_SELECTION_POLICY
                info["assumption"] = "No unconverted profile/color-tag conflicts; preview interpreted as encoded sRGB"
                kind = "DNG preview"
            elif path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                rgb, kind, info = _load_raster(path)
            else:
                raise ImageLoadError("Supported inputs are JPG/JPEG, DNG JPEG previews, and RGB PNG8")
            rgb = _rgb8(rgb)
            preview = Image.fromarray(rgb)
            preview.thumbnail((PREVIEW_EDGE, PREVIEW_EDGE), Image.Resampling.LANCZOS)
            return ImageDocument(path, rgb, np.asarray(preview), kind, info, rgb.shape[1], rgb.shape[0])
        except ImageLoadError:
            raise
        except dng_preview.PreviewError as error:
            if "ICC" in str(error) or "color" in str(error).lower():
                raise ImageLoadError(
                    "DNG preview has unsupported/unconverted color tags or ICC; this app does not "
                    "acknowledge them automatically. Supply a color-converted sRGB JPEG instead. " + str(error)) from error
            raise ImageLoadError(f"DNG preview rejected: {error}") from error
        except (OSError, ValueError, SyntaxError, struct.error, UserWarning, Image.DecompressionBombError) as error:
            raise ImageLoadError(f"Cannot safely load {path.name}: {error}") from error

    def render(self, rgb: np.ndarray, look_id: str, strength: float | None = None, *,
               progress: Progress | None = None, cancel: threading.Event | None = None,
               color_filter: str | None = None) -> np.ndarray:
        """Render RGB8 with the selected catalog recipe and one final quantizer.

        Steve preserves its observed decimal endpoint mix. Greg's candidate
        endpoints are mixed in P3 before conversion back to sRGB. Single-tone
        source blends happen in encoded sRGB *after* output-space conversion,
        always against the unfiltered source; 0 returns exact original pixels.
        Optional permitted attachments are sampled before both tone tables in
        the working domain. P3 conversion never clips extended floating values.
        The return is independent/writable; the source remains untouched.
        """
        source = _rgb8(rgb)
        look, amount = _look_strength(look_id, strength)
        binding = LOOK_BINDINGS[look.id]
        _report(progress, 0.0, cancel)
        primary, secondary, filter_table = self._tables(look.id, color_filter)
        _check_cancel(cancel)
        flat = source.reshape(-1, 3)
        output = np.empty(flat.shape, dtype=np.uint8)
        recipe = binding["recipe"]
        for start in range(0, len(flat), CHUNK_PIXELS):
            _check_cancel(cancel)
            end = min(start + CHUNK_PIXELS, len(flat))
            if recipe == "identity" or (recipe == "source_to_primary" and amount == 0):
                output[start:end] = flat[start:end]
            else:
                source_rgb = flat[start:end].astype(np.float64) / 255
                coordinates = source_rgb
                if binding["input_space"] != "srgb":
                    coordinates = color_spaces.convert_rgb(
                        coordinates, "srgb", binding["input_space"], clip=False)
                if filter_table is not None:
                    coordinates = renderer.sample_texture(filter_table, coordinates)
                if recipe == "secondary_to_primary":
                    values = renderer.mix_tables(secondary, primary, coordinates, amount / 100)
                else:
                    values = renderer.sample_texture(primary, coordinates)
                if binding["output_space"] != "srgb":
                    values = color_spaces.convert_rgb(values, binding["output_space"], "srgb", clip=False)
                if recipe == "source_to_primary" and amount != 100:
                    # Do not clamp, convert back to bytes, or blend in P3 here.
                    # Bypass this at 100 to keep old validated floating math.
                    values = source_rgb + (values - source_rgb) * (amount / 100)
                output[start:end] = renderer.quantize_u8(values)
            _report(progress, end / len(flat), cancel)
        return output.reshape(source.shape)

    def export(self, doc: ImageDocument, path: str | os.PathLike, look_id: str,
               strength: float | None = None, *, format: str | None = None, quality: int = 95,
               progress: Progress | None = None, cancel: threading.Event | None = None,
               color_filter: str | None = None) -> dict:
        """Write exactly one *new* RGB8 PNG/JPEG, with sRGB and clean EXIF.

        No parent directories or sidecars are created. Existing destinations,
        symlinks (including dangling ones), hardlinks and input aliases are
        refused. Cleanup only removes our newly created inode if still present.
        """
        if not isinstance(doc, ImageDocument):
            raise TypeError("doc must be an ImageDocument")
        target = Path(path).expanduser().absolute()
        extensions = {".png": "PNG", ".jpg": "JPEG", ".jpeg": "JPEG"}
        inferred = extensions.get(target.suffix.lower())
        chosen = inferred if format is None else str(format).upper()
        if chosen == "JPG":
            chosen = "JPEG"
        if chosen not in ("PNG", "JPEG") or inferred != chosen:
            raise ExportError("Output must be a new .png/.jpg/.jpeg with matching PNG/JPEG format")
        if type(quality) is not int or not 1 <= quality <= 100:
            raise ValueError("JPEG quality must be an integer in 1..100")
        look, amount = _look_strength(look_id, strength)
        _check_cancel(cancel)
        if target.resolve() == doc.path.resolve():
            raise ExportError("Output must not equal or alias the input")
        if os.path.lexists(target):
            raise ExportError(f"Refusing existing output or symlink: {target}")
        rendered = self.render(doc.rgb, look.id, amount, color_filter=color_filter,
                               progress=(lambda p: _report(progress, p * 0.9, cancel)), cancel=cancel)
        _check_cancel(cancel)
        owned = None
        try:
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0), 0o600)
            try:
                owned = os.fstat(fd)
                handle = os.fdopen(fd, "wb")
            except BaseException:
                os.close(fd)
                raise
            with handle:
                _check_cancel(cancel)
                exif = Image.Exif()
                exif[274] = 1
                exif[305] = "Local Looks (independent RGB reference)"
                exif[34665] = {40961: 1}
                profile = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
                options = {"icc_profile": profile, "exif": exif}
                if chosen == "JPEG":
                    options.update(quality=quality, subsampling=0)
                else:
                    text = PngImagePlugin.PngInfo()
                    text.add_text("Software", SOFTWARE)
                    options["pnginfo"] = text
                Image.fromarray(rendered).save(handle, format=chosen, **options)
                _check_cancel(cancel)
                handle.flush()
                os.fsync(handle.fileno())
            _report(progress, 1.0, cancel)
        except BaseException as error:
            if owned is not None:
                try:
                    current = target.lstat()
                    if (current.st_dev, current.st_ino) == (owned.st_dev, owned.st_ino):
                        target.unlink()
                except FileNotFoundError:
                    pass
            if isinstance(error, OSError) and not isinstance(error, EngineError):
                raise ExportError(f"Could not create new output {target.name}: {error}") from error
            raise
        return {"path": str(target), "format": chosen, "look_id": look.id, "strength": amount,
                "strength_rule": LOOK_STRENGTH_RULES[look.id],
                "recipe": LOOK_BINDINGS[look.id]["recipe"],
                "working_domain": LOOK_BINDINGS[look.id]["input_space"],
                "output_domain": "srgb", "color_filter": color_filter,
                "preview": look_preview(look.id, amount, color_filter),
                "width": doc.width, "height": doc.height, "mode": "RGB", "color_space": "sRGB",
                "orientation": 1, "gps_included": False, "source_kind": doc.source_kind,
                "software": SOFTWARE}
