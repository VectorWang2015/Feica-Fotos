#!/usr/bin/env python3
"""Build Local Looks for THIS OS only, using a private allowlisted staging tree.

No installation, downloads, cross compilation, photo input, or publication.
PyInstaller runs only on explicit invocation without --dry-run. Vendor LUTs are
excluded unless --with-local-resources is explicitly supplied by their holder.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
APP_NAME = "LocalLooks"
SOURCE_FILES = (
    "apps/local_looks/__init__.py",
    "apps/local_looks/__main__.py",
    "apps/local_looks/ui.py",
    "apps/local_looks/engine.py",
    "apps/local_looks/look_info.py",
    "apps/local_looks/glass.py",
    "apps/local_looks/catalog.py",
    "reproduction/ios_looks/color_spaces.py",
    "reproduction/ios_looks/renderer.py",
    "reproduction/ios_looks/dng_preview.py",
)
# Approved original icon assets only, not reference photos or rejected studies.
ICON_FILES = tuple('apps/local_looks/assets/icon-'+str(size)+'.png' for size in (16,24,32,48,64,128,256,512)) + (
    'apps/local_looks/assets/local-looks.ico', 'apps/local_looks/assets/icon-c.svg', 'apps/local_looks/assets/icon-manifest.json')
APP_DATA_FILES = ('apps/local_looks/assets/official-look-descriptions.json', 'apps/local_looks/assets/look-catalog.json')
# Resource names come only from the explicit bounded runtime catalog, never a directory glob.
def resource_names(catalog_bytes):
    catalog=json.loads(catalog_bytes)
    names=set()
    for look in catalog['looks']:
        for key in ('primary_cube','secondary_cube'):
            if look.get(key):names.add(look[key]['filename'])
    for item in catalog['color_filters']:names.add(item['cube']['filename'])
    for name in names:
        if not isinstance(name,str) or not name.endswith('.cube') or name.startswith('.') or chr(92) in name or chr(0) in name or '/' in name or ':' in name:
            raise ValueError('Catalog resource must be a plain .cube filename')
    if len(names)>64:raise ValueError('Catalog exceeds64 explicit resource entries')
    return tuple(sorted(names))
NOTICE = """Local Looks — private offline research build

Independent local software. Not an official Leica product; no affiliation,
endorsement, or authorization is implied. DNG input uses an embedded JPEG
preview, not RAW development. Only the existing RGB Look stage is reproduced.

Vendor LUTs are NOT included by default. If this build contains look-resources,
the person creating it explicitly supplied the catalog's local tables for private use.
Possession is not a redistribution license. Do not publicly publish these
resources or this resource-bearing bundle without resolving all rights.
This builder does not download, install, upload, or publish anything.

Third-party Python, Qt/PySide6, NumPy, Pillow and PyInstaller components retain
their own licenses. A public distribution requires a separate license review
and any required notices/source offers; this notice is not a license grant.
"""


class BuildError(RuntimeError):
    pass


def read_regular(path: Path, *, max_bytes: int) -> bytes:
    """Copy only an explicitly named regular file, not a link/device/directory."""
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or stat.S_ISLNK(before.st_mode):
        raise BuildError(f"Refusing non-regular file or symlink: {path}")
    if getattr(before, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400):
        raise BuildError(f"Refusing Windows reparse-point file: {path}")
    if not 0 < before.st_size <= max_bytes:
        raise BuildError(f"File is empty or exceeds {max_bytes} bytes: {path}")
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0))
    with os.fdopen(fd, "rb") as handle:
        opened = os.fstat(handle.fileno())
        if not stat.S_ISREG(opened.st_mode):
            raise BuildError(f"File changed to a non-regular file: {path}")
        content = handle.read(max_bytes + 1)
        after = os.fstat(handle.fileno())
    stamp = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns)
    if stamp(before) != stamp(opened) or stamp(opened) != stamp(after) or len(content) != before.st_size:
        raise BuildError(f"File changed while reading: {path}")
    return content


def load_inputs(resource_dir: Path | None) -> tuple[dict[str, bytes], dict[str, bytes]]:
    sources = {name: read_regular(ROOT / name, max_bytes=4 * 1024 * 1024) for name in SOURCE_FILES + ICON_FILES + APP_DATA_FILES}
    resources = {}
    if resource_dir is not None:
        folder = resource_dir.expanduser().resolve(strict=True)
        if not folder.is_dir():
            raise BuildError("--with-local-resources must name a directory containing the catalog .cube files")
        for name in resource_names(sources['apps/local_looks/assets/look-catalog.json']):
            # Explicit catalog basenames only, never siblings, subdirectories, ICCs or images.
            resources[name] = read_regular(folder / name, max_bytes=64 * 1024 * 1024)
    return sources, resources


def digest_records(files: dict[str, bytes]) -> list[dict]:
    return [{"name": name, "bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            for name, content in files.items()]


def dependency_versions() -> dict[str, str]:
    if sys.version_info[:2] != (3, 12):
        raise BuildError("Use Python 3.12 in the project .venv; this build target does not support other Python versions")
    if sys.prefix == sys.base_prefix:
        raise BuildError("Use a virtual environment, not a global Python installation")
    versions = {}
    for package in ("PyInstaller", "PySide6-Essentials", "numpy", "Pillow"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError as error:
            raise BuildError(
                f"Missing {package}. Explicitly run the project .venv Python with "
                "-m pip install -r requirements-app.txt -r requirements-build.txt. "
                "This script never installs dependencies."
            ) from error
    if versions["PySide6-Essentials"] != "6.11.2" or versions["numpy"] != "1.26.4":
        raise BuildError("Runtime versions differ from requirements-app.txt; install its pinned dependencies first")
    pillow = tuple(int(part) for part in versions["Pillow"].split(".")[:2])
    if not (pillow >= (11, 3) and pillow < (13, 0)):
        raise BuildError("Pillow must satisfy >=11.3,<13")
    return versions


def stage_sources(stage: Path, sources: dict[str, bytes], resources: dict[str, bytes]) -> Path:
    for relative, content in sources.items():
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    # Explicit regular packages prevent imports resolving to a research namespace.
    for relative in ("apps/__init__.py", "reproduction/__init__.py", "reproduction/ios_looks/__init__.py"):
        (stage / relative).write_text('"""Local Looks isolated build package."""\n', encoding="utf-8")
    entry = stage / "local_looks_entry.py"
    entry.write_text(
        "from apps.local_looks.ui import main\n"
        "if __name__ == '__main__':\n"
        "    raise SystemExit(main())\n", encoding="utf-8")
    if resources:
        folder = stage / "look-resources"
        folder.mkdir()
        for name, content in resources.items():
            (folder / name).write_bytes(content)
    return entry


def pyinstaller_command(stage: Path, entry: Path, with_resources: bool) -> list[str]:
    command = [sys.executable, "-m", "PyInstaller", "--name", APP_NAME,
               "--onedir", "--contents-directory", "_internal", "--noupx",
               "--noconfirm", "--clean", "--paths", str(stage),
               "--specpath", str(stage / "spec"),
               "--workpath", str(stage / "work"),
               "--distpath", str(stage / "dist")]
    command.extend(('--add-data', str(stage/'apps/local_looks/assets') + os.pathsep + 'apps/local_looks/assets'))
    if sys.platform == "win32":
        command.extend(('--windowed','--icon',str(stage/'apps/local_looks/assets/local-looks.ico')))
    for name in ("tkinter", "matplotlib", "PySide6.QtQml", "PySide6.QtQuick", "PySide6.QtWebEngineCore"):
        command.extend(("--exclude-module", name))
    if with_resources:
        command.extend(("--add-data", str(stage / "look-resources") + os.pathsep + "look-resources"))
    command.append(str(entry))
    return command


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-local-resources", type=Path, metavar="PATH",
                        help="Explicit private-use opt-in: include only the catalog-allowlisted LUTs from PATH")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate the source/resource allowlists and print the plan; do not invoke PyInstaller")
    args = parser.parse_args(argv)
    try:
        if sys.platform not in ("linux", "win32"):
            raise BuildError("This builder supports native Ubuntu/Linux and Windows only")
        target = ROOT / "dist" / APP_NAME
        sources, resources = load_inputs(args.with_local_resources)
        plan = {
            "output": str(target), "build_os": platform.system(), "architecture": platform.machine(),
            "mode": "onedir", "cross_compile": False, "public_release_authorized": False,
            "source_allowlist": digest_records(sources), "vendor_resources": digest_records(resources),
            "resource_policy": "explicit private local opt-in" if resources else "excluded by default",
            "excluded_repository_data": ["samples", "IPA", "firmware", "ICC files", "private photos", "research reports"],
        }
        if args.dry_run:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            print("Dry run only. Dependencies and native bundle execution are not verified.")
            return 0
        if os.path.lexists(target):
            raise BuildError(f"Output already exists; move it aside yourself before rebuilding: {target}")
        versions = dependency_versions()
        print(f"Building on {platform.system()} {platform.machine()} for this platform only.")
        if resources:
            print("PRIVATE RESOURCE BUILD: do not publicly redistribute the included vendor tables.")
        # Source staging and PyInstaller's CWD are outside the research checkout.
        # Never add ROOT via --paths or --add-data, and never collect_all the repo.
        with tempfile.TemporaryDirectory(prefix="local-looks-build-") as temporary:
            stage = Path(temporary)
            entry = stage_sources(stage, sources, resources)
            environment = os.environ.copy()
            for key in ("PYTHONPATH", "PYTHONHOME"):
                environment.pop(key, None)
            environment["PYTHONNOUSERSITE"] = "1"
            environment["PYINSTALLER_CONFIG_DIR"] = str(stage / "pyinstaller-cache")
            command = pyinstaller_command(stage, entry, bool(resources))
            print("Running:", subprocess.list2cmdline(command), flush=True)
            subprocess.run(command, cwd=stage, env=environment, check=True)
            bundle = stage / "dist" / APP_NAME
            executable = bundle / (APP_NAME + (".exe" if sys.platform == "win32" else ""))
            if not executable.is_file():
                raise BuildError("PyInstaller returned without the expected executable")
            # Extra defense: the only repository .cube data in the output is this exact opt-in set.
            actual_tables = sorted(p.relative_to(bundle).as_posix() for p in bundle.rglob("*.cube"))
            expected_tables = sorted("_internal/look-resources/" + name for name in resources)
            if actual_tables != expected_tables:
                raise BuildError("Bundle LUT inventory differs from the explicit allowlist; output withheld")
            build_info = {**plan, "created_utc": datetime.now(timezone.utc).isoformat(),
                          "python": platform.python_version(), "dependencies": versions,
                          "windows_execution_verified": False,
                          "notes": "Build metadata is not a runtime/platform validation or redistribution license."}
            # Avoid embedding the developer's absolute checkout path in public metadata.
            build_info["output"] = "dist/LocalLooks"
            (bundle / "BUILD-INFO.json").write_text(json.dumps(build_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            (bundle / "PRIVATE-USE-NOTICE.txt").write_text(NOTICE, encoding="utf-8")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.mkdir()  # Exclusive reservation: never overwrite an existing output tree.
            # Retain symlinks made by PyInstaller for Qt/shared-library loader behavior.
            shutil.copytree(bundle, target, dirs_exist_ok=True, symlinks=True)
        print(f"Created {target}. Keep the executable and _internal directory together.")
        print("No runtime validation was performed by this script. Test on the target OS before use.")
        return 0
    except (BuildError, OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Build stopped: {error}", file=sys.stderr)
        print("No dependencies were installed and nothing was published.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
