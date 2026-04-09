#!/usr/bin/env python3

# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Cross-platform Python dependency installer for Unsloth Studio.

Called by both setup.sh (Linux / WSL) and setup.ps1 (Windows) after the
virtual environment is already activated.  Expects `pip` and `python` on
PATH to point at the venv.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"
IS_MAC_INTEL = IS_MACOS and platform.machine() == "x86_64"

# ── ROCm / AMD GPU support ─────────────────────────────────────────────────────
# Mapping from detected ROCm (major, minor) to the best PyTorch wheel tag on
# download.pytorch.org.  Entries are checked newest-first (>=).
# ROCm 7.2 only has torch 2.11.0 on download.pytorch.org, which exceeds the
# current torch upper bound (<2.11.0).  Fall back to rocm7.1 (torch 2.10.0).
# TODO: uncomment rocm7.2 when torch upper bound is bumped to >=2.11.0
_ROCM_TORCH_INDEX: dict[tuple[int, int], str] = {
    # (7, 2): "rocm7.2",  # torch 2.11.0 -- requires torch>=2.11
    (7, 1): "rocm7.1",
    (7, 0): "rocm7.0",
    (6, 4): "rocm6.4",
    (6, 3): "rocm6.3",
    (6, 2): "rocm6.2",
    (6, 1): "rocm6.1",
    (6, 0): "rocm6.0",
}
_PYTORCH_WHL_BASE = "https://download.pytorch.org/whl"


def _detect_rocm_version() -> tuple[int, int] | None:
    """Return (major, minor) of the installed ROCm stack, or None."""
    # Check /opt/rocm/.info/version or ROCM_PATH equivalent
    rocm_root = os.environ.get("ROCM_PATH") or "/opt/rocm"
    for path in (
        os.path.join(rocm_root, ".info", "version"),
        os.path.join(rocm_root, "lib", "rocm_version"),
    ):
        try:
            with open(path) as fh:
                parts = fh.read().strip().split("-")[0].split(".")
            # Explicit length guard avoids relying on the broad except
            # below to swallow IndexError when the version file contains
            # a single component (e.g. "6\n" on a partial install).
            if len(parts) >= 2:
                return int(parts[0]), int(parts[1])
        except Exception:
            pass

    # Try amd-smi version (outputs "... | ROCm version: X.Y.Z")
    amd_smi = shutil.which("amd-smi")
    if amd_smi:
        try:
            result = subprocess.run(
                [amd_smi, "version"],
                stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                text = True,
                timeout = 5,
            )
            if result.returncode == 0:
                import re

                m = re.search(r"ROCm version:\s*(\d+)\.(\d+)", result.stdout)
                if m:
                    return int(m.group(1)), int(m.group(2))
        except Exception:
            pass

    # Try hipconfig --version (outputs bare version like "6.3.21234.2").
    # Use text=True for consistency with every other subprocess call in
    # this file and with _detect_host_rocm_version() in
    # studio/install_llama_prebuilt.py; a manual .decode() can raise on
    # non-UTF-8 builds and would not report any stderr.
    hipconfig = shutil.which("hipconfig")
    if hipconfig:
        try:
            result = subprocess.run(
                [hipconfig, "--version"],
                stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                text = True,
                timeout = 5,
            )
            if result.returncode == 0:
                raw = (result.stdout or "").strip().split("\n")[0]
                parts = raw.split(".")
                if (
                    len(parts) >= 2
                    and parts[0].isdigit()
                    and parts[1].split("-")[0].isdigit()
                ):
                    return int(parts[0]), int(parts[1].split("-")[0])
        except Exception:
            pass

    # Distro package-manager fallbacks. Package-managed ROCm installs can
    # expose GPUs via rocminfo / amd-smi but still lack /opt/rocm/.info/version
    # and hipconfig, so probe dpkg (Debian/Ubuntu) and rpm (RHEL/Fedora/SUSE)
    # for the rocm-core package version. Matches the chain in
    # install.sh::get_torch_index_url so `unsloth studio update` behaves
    # the same as a fresh `curl | sh` install.
    import re as _re_pkg

    for cmd in (
        ["dpkg-query", "-W", "-f=${Version}\n", "rocm-core"],
        ["rpm", "-q", "--qf", "%{VERSION}\n", "rocm-core"],
    ):
        exe = shutil.which(cmd[0])
        if not exe:
            continue
        try:
            result = subprocess.run(
                [exe, *cmd[1:]],
                stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                text = True,
                timeout = 5,
            )
        except Exception:
            continue
        if result.returncode != 0 or not result.stdout.strip():
            continue
        raw = result.stdout.strip()
        # dpkg can prepend an epoch ("1:6.3.0-1"); strip it before parsing.
        raw = _re_pkg.sub(r"^\d+:", "", raw)
        m = _re_pkg.match(r"(\d+)[.-](\d+)", raw)
        if m:
            return int(m.group(1)), int(m.group(2))

    return None


def _has_rocm_gpu() -> bool:
    """Return True only if an actual AMD GPU is visible (not just ROCm tools installed)."""
    import re

    for cmd, check_fn in (
        # rocminfo: look for "Name: gfxNNNN" with nonzero first digit (gfx000 is the CPU agent)
        (["rocminfo"], lambda out: bool(re.search(r"gfx[1-9]", out.lower()))),
        # amd-smi list: require a data row with a GPU index. amd-smi
        # ships three format variants across versions:
        #   "GPU: 0"   (colon separator)
        #   "GPU[0]"   (bracket wrapper)
        #   "GPU 0"    (space then digit, no separator)
        # The plain "GPU" header (no following digit) is still rejected.
        (
            ["amd-smi", "list"],
            lambda out: bool(re.search(r"(?im)^gpu\s*(?:[:\[]\s*|\s+)\d", out)),
        ),
    ):
        exe = shutil.which(cmd[0])
        if not exe:
            continue
        try:
            result = subprocess.run(
                [exe, *cmd[1:]],
                stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                text = True,
                timeout = 10,
            )
        except Exception:
            continue
        if result.returncode == 0 and result.stdout.strip():
            if check_fn(result.stdout):
                return True
    return False


def _has_radeon_gpu() -> bool:
    """Return True when a consumer AMD Radeon GPU is detected.

    Mirrors the `_is_radeon_gpu` shell helper in install.sh: prefer the
    rocminfo ``Marketing Name`` line, fall back to ``amd-smi static``.
    """
    import re as _re_radeon

    rocminfo = shutil.which("rocminfo")
    if rocminfo:
        try:
            result = subprocess.run(
                [rocminfo],
                stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                text = True,
                timeout = 10,
            )
        except Exception:
            result = None
        if (
            result is not None
            and result.returncode == 0
            and _re_radeon.search(r"(?m)^\s*Marketing Name:.*Radeon", result.stdout)
        ):
            return True
    amd_smi = shutil.which("amd-smi")
    if amd_smi:
        try:
            result = subprocess.run(
                [amd_smi, "static"],
                stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                text = True,
                timeout = 10,
            )
        except Exception:
            result = None
        if (
            result is not None
            and result.returncode == 0
            and "radeon" in (result.stdout or "").lower()
        ):
            return True
    return False


def _radeon_fetch_listing(base_url: str) -> str | None:
    """Fetch the Radeon manylinux directory listing, capped at 10 MiB.

    Returns the HTML body (decoded as UTF-8 with replacement) or None on
    network / size failure. A response larger than 10 MiB is rejected
    instead of streamed to prevent a pathological listing from
    exhausting memory during regex parsing.
    """
    try:
        req = urllib.request.Request(base_url, method = "GET")
        with urllib.request.urlopen(req, timeout = 20) as response:
            body = response.read(10 * 1024 * 1024 + 1)
    except Exception:
        return None
    if not body or len(body) > 10 * 1024 * 1024:
        return None
    try:
        return body.decode("utf-8", errors = "replace")
    except Exception:
        return None


def _parse_radeon_wheel_version(url: str) -> tuple[str, str] | None:
    """Return ``(major, minor)`` for a Radeon wheel URL, else None.

    Mirrors the `_torch_ver`/`_tv_ver`/`_ta_ver` extraction in install.sh.
    Radeon wheel hrefs are percent-encoded (``torch-2.10.0%2Brocm7.2.0...``)
    so we URL-decode the filename before parsing and then take the
    first ``major.minor`` run as the public version.
    """
    import re as _re_ver
    import urllib.parse as _urlparse

    name = url.rsplit("/", 1)[-1]
    name = _urlparse.unquote(name)
    m = _re_ver.match(r"^[A-Za-z_]+-([0-9]+)\.([0-9]+)", name)
    if not m:
        return None
    return m.group(1), m.group(2)


def _radeon_wheels_compatible(
    torch_url: str, tv_url: str, ta_url: str
) -> bool:
    """Return True when the three Radeon wheels form a compatible trio.

    Ports the `_radeon_versions_match` shell guard:

      - torch.major == torchaudio.major
      - torch.minor == torchaudio.minor
      - torchvision.major == 0
      - torchvision.minor == torch.minor + 15

    The Radeon repo publishes multiple generations simultaneously so
    picking the highest-version wheel per package can assemble a
    mismatched trio (torch 2.9 + torchvision 0.23 + torchaudio 2.9).
    """
    torch_ver = _parse_radeon_wheel_version(torch_url)
    tv_ver = _parse_radeon_wheel_version(tv_url)
    ta_ver = _parse_radeon_wheel_version(ta_url)
    if torch_ver is None or tv_ver is None or ta_ver is None:
        return False
    torch_major, torch_minor = torch_ver
    ta_major, ta_minor = ta_ver
    tv_major, tv_minor = tv_ver
    if torch_major != ta_major:
        return False
    if torch_minor != ta_minor:
        return False
    if tv_major != "0":
        return False
    try:
        expected_tv_minor = int(torch_minor) + 15
    except ValueError:
        return False
    try:
        return int(tv_minor) == expected_tv_minor
    except ValueError:
        return False


def _pick_radeon_wheel_url(
    listing: str, base_url: str, package: str, python_tag: str
) -> str | None:
    """Pick the newest wheel in ``listing`` for ``package`` + ``python_tag``.

    Mirrors the shell `_pick_radeon_wheel` helper: parse ``href`` values,
    require an exact ``PACKAGE-`` prefix, a ``-{python_tag}-`` segment,
    and a linux x86_64 platform suffix. Pads each numeric component of
    the version for a plain lexical comparison so the newest release
    wins. Returns a fully-qualified URL (resolving relative hrefs
    against ``base_url``) or None.
    """
    import re as _re_rad

    best_pad: str | None = None
    best_href: str | None = None
    prefix = f"{package}-"
    version_re = _re_rad.compile(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?")
    for match in _re_rad.finditer(r'href="([^"]+)"', listing):
        raw_href = match.group(1)
        base_name = raw_href.rsplit("/", 1)[-1]
        base_name = _re_rad.sub(r"[?#].*", "", base_name)
        if not base_name.startswith(prefix):
            continue
        if f"-{python_tag}-" not in base_name:
            continue
        if not _re_rad.search(r"x86_64\.whl$", base_name):
            continue
        ver_match = version_re.search(base_name)
        if ver_match is None:
            continue
        pad = "".join(f"{int(part):08d}" for part in ver_match.group(0).split("."))
        if best_pad is None or pad > best_pad:
            best_pad = pad
            best_href = raw_href
    if best_href is None:
        return None
    # Reject plaintext / flag-injection hrefs; resolve relative paths
    # against the known-https Radeon base URL.
    if best_href.startswith("https://"):
        return best_href
    if best_href.startswith("http://") or best_href.startswith("-"):
        return None
    if "://" in best_href:
        return None
    base_stripped = base_url.rstrip("/")
    if best_href.startswith("/"):
        return f"{base_stripped}{best_href}"
    return f"{base_stripped}/{best_href}"


def _radeon_wheel_index(ver: str) -> str | None:
    """Return the Radeon manylinux wheel index URL for the given ROCm
    version string (e.g. ``"7.2.1"`` or ``"7.2"``) if it is reachable.

    install.sh prefers `repo.radeon.com` wheels on Radeon hosts so the
    Studio update path should do the same; otherwise a fresh install
    ends up on Radeon wheels but a subsequent `unsloth studio update`
    silently overwrites them with the generic PyTorch ROCm wheels.
    """
    base = f"https://repo.radeon.com/rocm/manylinux/rocm-rel-{ver}/"
    try:
        req = urllib.request.Request(base, method = "HEAD")
        with urllib.request.urlopen(req, timeout = 10):
            return base
    except Exception:
        return None


def _detect_rocm_full_version() -> str | None:
    """Return the host ROCm version as ``"X.Y"`` or ``"X.Y.Z"``.

    Used by the Radeon repo resolver which needs the full patch
    version (``7.2.1`` vs ``7.2``) to address the correct directory
    under ``https://repo.radeon.com/rocm/manylinux/``.
    """
    import re as _re_full

    rocm_root = os.environ.get("ROCM_PATH") or "/opt/rocm"
    for path in (
        os.path.join(rocm_root, ".info", "version"),
        os.path.join(rocm_root, "lib", "rocm_version"),
    ):
        try:
            with open(path) as fh:
                raw = fh.read().strip()
        except Exception:
            continue
        m = _re_full.search(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?", raw)
        if m:
            return m.group(0)

    amd_smi = shutil.which("amd-smi")
    if amd_smi:
        try:
            result = subprocess.run(
                [amd_smi, "version"],
                stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                text = True,
                timeout = 5,
            )
        except Exception:
            result = None
        if result is not None and result.returncode == 0:
            m = _re_full.search(
                r"ROCm version:\s*([0-9]+\.[0-9]+(?:\.[0-9]+)?)", result.stdout
            )
            if m:
                return m.group(1)

    hipconfig = shutil.which("hipconfig")
    if hipconfig:
        try:
            result = subprocess.run(
                [hipconfig, "--version"],
                stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                text = True,
                timeout = 5,
            )
        except Exception:
            result = None
        if result is not None and result.returncode == 0:
            m = _re_full.search(
                r"[0-9]+\.[0-9]+(?:\.[0-9]+)?", (result.stdout or "").split("\n")[0]
            )
            if m:
                return m.group(0)

    for cmd in (
        ["dpkg-query", "-W", "-f=${Version}\n", "rocm-core"],
        ["rpm", "-q", "--qf", "%{VERSION}\n", "rocm-core"],
    ):
        exe = shutil.which(cmd[0])
        if not exe:
            continue
        try:
            result = subprocess.run(
                [exe, *cmd[1:]],
                stdout = subprocess.PIPE,
                stderr = subprocess.DEVNULL,
                text = True,
                timeout = 5,
            )
        except Exception:
            continue
        if result.returncode != 0 or not (result.stdout or "").strip():
            continue
        raw = result.stdout.strip()
        # Strip Debian epoch like "1:6.3.0-1ubuntu1"
        raw = _re_full.sub(r"^[0-9]+:", "", raw)
        m = _re_full.search(r"[0-9]+\.[0-9]+(?:\.[0-9]+)?", raw)
        if m:
            return m.group(0)

    return None


def _has_usable_nvidia_gpu() -> bool:
    """Return True only when nvidia-smi exists AND reports at least one GPU.

    Tries two structured probes in order so older or stripped nvidia-smi
    builds that do not support ``-L`` are not misclassified as "no NVIDIA
    GPU":

      1. ``nvidia-smi -L``
      2. ``nvidia-smi --query-gpu=index --format=csv,noheader``

    A broken/stale ``nvidia-smi`` that fails both structured probes but
    still prints a banner with ``CUDA Version: X.Y`` is deliberately NOT
    treated as usable: on an AMD host that is the same output a dead
    driver emits, and we want to fall through to the ROCm branch.
    """
    import re as _re_nv

    exe = shutil.which("nvidia-smi")
    if not exe:
        return False

    # Probe 1: -L must produce at least one "GPU N:" line.
    try:
        result = subprocess.run(
            [exe, "-L"],
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            text = True,
            timeout = 10,
        )
    except Exception:
        result = None
    if result is not None and result.returncode == 0 and _re_nv.search(
        r"(?m)^GPU\s+\d+:", result.stdout or ""
    ):
        return True

    # Probe 2: --query-gpu=index must produce only non-negative integer
    # indices (optional trailing comma). A stale nvidia-smi that prints
    # the default banner for the flag it does not understand would
    # otherwise slip through a looser non-empty check.
    try:
        result = subprocess.run(
            [exe, "--query-gpu=index", "--format=csv,noheader"],
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            text = True,
            timeout = 10,
        )
    except Exception:
        return False
    if result.returncode != 0:
        return False
    saw_index = False
    for line in (result.stdout or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if not _re_nv.fullmatch(r"[0-9]+\s*,?", stripped):
            return False
        saw_index = True
    return saw_index


def _ensure_rocm_torch() -> None:
    """Reinstall torch with ROCm wheels when the venv received CPU-only torch.

    Runs only on Linux x86_64 hosts where an AMD GPU is present and the
    ROCm runtime is detectable (rocminfo / amd-smi / hipconfig /
    rocm-core package).  No-op when torch already links against HIP
    (ROCm), on Windows / macOS, on non-x86_64 Linux (PyTorch does not
    publish ROCm wheels for aarch64 / arm64), or on mixed AMD+NVIDIA
    hosts (NVIDIA takes precedence).
    Uses pip_install() to respect uv, constraints, and --python targeting.
    """
    # Explicit OS / architecture guards so the helper is safe to call
    # from any context -- PyTorch only publishes ROCm wheels for
    # linux_x86_64, so aarch64 / arm64 hosts must skip this repair path
    # instead of failing the update with a missing-wheel error.
    if IS_WINDOWS or IS_MACOS:
        return
    if platform.machine().lower() not in {"x86_64", "amd64"}:
        return
    # NVIDIA takes precedence on mixed hosts -- but only if an actual GPU is usable
    if _has_usable_nvidia_gpu():
        return
    # Rely on _has_rocm_gpu() (rocminfo / amd-smi GPU data rows) as the
    # authoritative "is this actually an AMD ROCm host?" signal. The old
    # gate required /opt/rocm or hipcc to exist, which breaks on
    # runtime-only ROCm installs (package-managed minimal installs,
    # Radeon software) that ship amd-smi/rocminfo without /opt/rocm or
    # hipcc, and leaves `unsloth studio update` unable to repair a
    # CPU-only venv on those systems.
    if not _has_rocm_gpu():
        return  # no AMD GPU visible

    ver = _detect_rocm_version()
    if ver is None:
        print("   ROCm detected but version unreadable -- skipping torch reinstall")
        return

    # Probe whether torch already links against HIP (ROCm is already working).
    # Do NOT skip for CUDA-only builds since they are unusable on AMD-only
    # hosts (the NVIDIA check above already handled mixed AMD+NVIDIA setups).
    try:
        probe = subprocess.run(
            [
                sys.executable,
                "-c",
                "import torch; print(getattr(torch.version,'hip','') or '')",
            ],
            stdout = subprocess.PIPE,
            stderr = subprocess.DEVNULL,
            timeout = 30,
        )
    except (OSError, subprocess.TimeoutExpired):
        probe = None
    has_hip_torch = (
        probe is not None
        and probe.returncode == 0
        and probe.stdout.decode().strip() != ""
    )

    rocm_torch_ready = has_hip_torch

    if not has_hip_torch:
        # On consumer Radeon hosts prefer the AMD-published
        # repo.radeon.com wheel set when it is reachable. This mirrors
        # the fresh-install path in install.sh so `unsloth studio
        # update` does not silently overwrite Radeon torch with the
        # generic PyTorch ROCm wheels.
        #
        # Important: we must pass explicit wheel URLs, not bare package
        # names with `--find-links`. `--find-links` is only an extra
        # source, so `uv`/`pip` may still resolve `torch` to a newer
        # PyPI release with a generic CUDA runtime. Parsing the Radeon
        # listing and picking wheels by CPython tag mirrors the
        # `_pick_radeon_wheel` helper in install.sh.
        radeon_installed = False
        if _has_radeon_gpu():
            radeon_full_ver = _detect_rocm_full_version()
            radeon_base: str | None = None
            if radeon_full_ver is not None:
                radeon_base = _radeon_wheel_index(radeon_full_ver)
                # AMD publishes both ``rocm-rel-X.Y.Z/`` and
                # ``rocm-rel-X.Y/``; if the full version is not reachable,
                # try the two-component form before giving up.
                if radeon_base is None and radeon_full_ver.count(".") >= 2:
                    radeon_base = _radeon_wheel_index(
                        ".".join(radeon_full_ver.split(".", 2)[:2])
                    )
            if radeon_base is not None:
                listing = _radeon_fetch_listing(radeon_base)
                python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
                torch_whl = (
                    _pick_radeon_wheel_url(listing, radeon_base, "torch", python_tag)
                    if listing
                    else None
                )
                tv_whl = (
                    _pick_radeon_wheel_url(
                        listing, radeon_base, "torchvision", python_tag
                    )
                    if listing
                    else None
                )
                ta_whl = (
                    _pick_radeon_wheel_url(
                        listing, radeon_base, "torchaudio", python_tag
                    )
                    if listing
                    else None
                )
                tri_whl = (
                    _pick_radeon_wheel_url(listing, radeon_base, "triton", python_tag)
                    if listing
                    else None
                )
                if not (torch_whl and tv_whl and ta_whl):
                    print(
                        "   Radeon repo listing did not contain a compatible "
                        f"wheel set for {python_tag}; falling back to generic ROCm "
                        "index"
                    )
                elif not _radeon_wheels_compatible(torch_whl, tv_whl, ta_whl):
                    # The Radeon repo publishes multiple generations in
                    # the same directory. Picking the highest version
                    # for each package independently can assemble a
                    # mismatched trio (torch 2.9 + torchvision 0.23 +
                    # torchaudio 2.9). Defer to the generic PyTorch
                    # ROCm index in that case instead of installing an
                    # incoherent wheel set.
                    print(
                        "   Radeon repo yielded a mismatched torch / torchvision / "
                        "torchaudio trio; falling back to generic ROCm index"
                    )
                else:
                    print(
                        f"   Radeon ROCm {radeon_full_ver} -- installing torch from "
                        f"{radeon_base}"
                    )
                    pip_args = [
                        f"ROCm torch (Radeon {radeon_full_ver})",
                        "--force-reinstall",
                        "--no-cache-dir",
                        "--find-links",
                        radeon_base,
                    ]
                    if tri_whl:
                        pip_args.append(tri_whl)
                    pip_args.extend([torch_whl, tv_whl, ta_whl])
                    pip_install(*pip_args, constrain = False)
                    radeon_installed = True
                    rocm_torch_ready = True
        if not radeon_installed:
            # Select best matching wheel tag (newest ROCm version <= installed)
            tag = next(
                (
                    t
                    for (maj, mn), t in sorted(_ROCM_TORCH_INDEX.items(), reverse = True)
                    if ver >= (maj, mn)
                ),
                None,
            )
            if tag is None:
                print(
                    f"   No PyTorch wheel for ROCm {ver[0]}.{ver[1]} -- "
                    f"skipping torch reinstall"
                )
            else:
                index_url = f"{_PYTORCH_WHL_BASE}/{tag}"
                print(f"   ROCm {ver[0]}.{ver[1]} -- installing torch from {index_url}")
                pip_install(
                    f"ROCm torch ({tag})",
                    "--force-reinstall",
                    "--no-cache-dir",
                    "torch>=2.4,<2.11.0",
                    "torchvision<0.26.0",
                    "torchaudio<2.11.0",
                    "--index-url",
                    index_url,
                    constrain = False,
                )
                rocm_torch_ready = True

    # Install bitsandbytes only when the venv has a ROCm-compatible torch
    # (either already present or just installed). Avoids leaving an AMD
    # bitsandbytes on top of a CPU/CUDA torch on hosts where the ROCm
    # runtime is older than any published torch wheel. Uses
    # --force-reinstall so an existing CPU/CUDA bitsandbytes is replaced
    # by the AMD build during upgrades.
    if rocm_torch_ready:
        pip_install(
            "bitsandbytes (AMD)",
            "--force-reinstall",
            "--no-cache-dir",
            "bitsandbytes>=0.49.1",
            constrain = False,
        )


def _infer_no_torch() -> bool:
    """Determine whether to run in no-torch (GGUF-only) mode.

    Checks UNSLOTH_NO_TORCH env var first.  When unset, falls back to
    platform detection so that Intel Macs automatically use GGUF-only
    mode even when invoked from ``unsloth studio update`` (which does
    not inject the env var).
    """
    env = os.environ.get("UNSLOTH_NO_TORCH")
    if env is not None:
        return env.strip().lower() in ("1", "true")
    return IS_MAC_INTEL


NO_TORCH = _infer_no_torch()

# -- Verbosity control ----------------------------------------------------------
# By default the installer shows a minimal progress bar (one line, in-place).
# Set UNSLOTH_VERBOSE=1 in the environment to restore full per-step output:
#   CLI:        unsloth studio setup --verbose
#   Linux/Mac:  UNSLOTH_VERBOSE=1 ./studio/setup.sh
#   Windows:    $env:UNSLOTH_VERBOSE="1" ; .\studio\setup.ps1
VERBOSE: bool = os.environ.get("UNSLOTH_VERBOSE", "0") == "1"

# Progress bar state -- updated by _progress() as each install step runs.
# _TOTAL counts: pip-upgrade + 7 shared steps + triton (non-Windows) + local-plugin + finalize
# Update _TOTAL here if you add or remove install steps in install_python_stack().
_STEP: int = 0
_TOTAL: int = 0  # set at runtime in install_python_stack() based on platform

# -- Paths --------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REQ_ROOT = SCRIPT_DIR / "backend" / "requirements"
SINGLE_ENV = REQ_ROOT / "single-env"
CONSTRAINTS = SINGLE_ENV / "constraints.txt"
LOCAL_DD_UNSTRUCTURED_PLUGIN = (
    SCRIPT_DIR / "backend" / "plugins" / "data-designer-unstructured-seed"
)

# -- Unicode-safe printing ---------------------------------------------
# On Windows the default console encoding can be a legacy code page
# (e.g. CP1252) that cannot represent Unicode glyphs such as ✅ or ❌.
# _safe_print() gracefully degrades to ASCII equivalents so the
# installer never crashes just because of a status glyph.

_UNICODE_TO_ASCII: dict[str, str] = {
    "\u2705": "[OK]",  # ✅
    "\u274c": "[FAIL]",  # ❌
    "\u26a0\ufe0f": "[!]",  # ⚠️  (warning + variation selector)
    "\u26a0": "[!]",  # ⚠  (warning without variation selector)
}


def _safe_print(*args: object, **kwargs: object) -> None:
    """Drop-in print() replacement that survives non-UTF-8 consoles."""
    try:
        print(*args, **kwargs)
    except UnicodeEncodeError:
        # Stringify, then swap emoji for ASCII equivalents
        text = " ".join(str(a) for a in args)
        for uni, ascii_alt in _UNICODE_TO_ASCII.items():
            text = text.replace(uni, ascii_alt)
        # Final fallback: replace any remaining unencodable chars
        print(
            text.encode(sys.stdout.encoding or "ascii", errors = "replace").decode(
                sys.stdout.encoding or "ascii", errors = "replace"
            ),
            **kwargs,
        )


# ── Color support ──────────────────────────────────────────────────────
# Same logic as startup_banner: NO_COLOR disables, FORCE_COLOR or TTY enables.


def _stdout_supports_color() -> bool:
    """True if we should emit ANSI colors (matches startup_banner)."""
    if os.environ.get("NO_COLOR", "").strip():
        return False
    if os.environ.get("FORCE_COLOR", "").strip():
        return True
    try:
        if not sys.stdout.isatty():
            return False
    except (AttributeError, OSError, ValueError):
        return False
    if IS_WINDOWS:
        try:
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_ulong()
            kernel32.GetConsoleMode(handle, ctypes.byref(mode))
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except (ImportError, AttributeError, OSError):
            return False
    return True


_HAS_COLOR = _stdout_supports_color()


# Column layout — matches setup.sh step() helper:
#   2-space indent, 15-char label (dim), then value.
_LABEL = "deps"
_COL = 15


def _green(msg: str) -> str:
    return f"\033[38;5;108m{msg}\033[0m" if _HAS_COLOR else msg


def _cyan(msg: str) -> str:
    return f"\033[96m{msg}\033[0m" if _HAS_COLOR else msg


def _red(msg: str) -> str:
    return f"\033[91m{msg}\033[0m" if _HAS_COLOR else msg


def _dim(msg: str) -> str:
    return f"\033[38;5;245m{msg}\033[0m" if _HAS_COLOR else msg


def _title(msg: str) -> str:
    return f"\033[38;5;150m{msg}\033[0m" if _HAS_COLOR else msg


_RULE = "\u2500" * 52


def _step(label: str, value: str, color_fn = None) -> None:
    """Print a single step line in the column format."""
    if color_fn is None:
        color_fn = _green
    padded = label[:_COL]
    print(f"  {_dim(padded)}{' ' * (_COL - len(padded))}{color_fn(value)}")


def _progress(label: str) -> None:
    """Print an in-place progress bar aligned to the step column layout."""
    global _STEP
    _STEP += 1
    if VERBOSE:
        return
    width = 20
    filled = int(width * _STEP / _TOTAL)
    bar = "=" * filled + "-" * (width - filled)
    pad = " " * (_COL - len(_LABEL))
    end = "\n" if _STEP >= _TOTAL else ""
    sys.stdout.write(
        f"\r  {_dim(_LABEL)}{pad}[{bar}] {_STEP:2}/{_TOTAL}  {label:<20}{end}"
    )
    sys.stdout.flush()


def run(
    label: str, cmd: list[str], *, quiet: bool = True
) -> subprocess.CompletedProcess[bytes]:
    """Run a command; on failure print output and exit."""
    if VERBOSE:
        _step(_LABEL, f"{label}...", _dim)
    result = subprocess.run(
        cmd,
        stdout = subprocess.PIPE if quiet else None,
        stderr = subprocess.STDOUT if quiet else None,
    )
    if result.returncode != 0:
        _step("error", f"{label} failed (exit code {result.returncode})", _red)
        if result.stdout:
            print(result.stdout.decode(errors = "replace"))
        sys.exit(result.returncode)
    return result


# Packages to skip on Windows (require special build steps)
WINDOWS_SKIP_PACKAGES = {"open_spiel", "triton_kernels"}

# Packages to skip when torch is unavailable (Intel Mac GGUF-only mode).
# These packages either *are* torch extensions or have unconditional
# ``Requires-Dist: torch`` in their published metadata, so installing
# them would pull torch back into the environment.
NO_TORCH_SKIP_PACKAGES = {
    "torch-stoi",
    "timm",
    "torchcodec",
    "torch-c-dlpack-ext",
    "openai-whisper",
    "transformers-cfg",
}

# -- uv bootstrap ------------------------------------------------------

USE_UV = False  # Set by _bootstrap_uv() at the start of install_python_stack()
UV_NEEDS_SYSTEM = False  # Set by _bootstrap_uv() via probe


def _bootstrap_uv() -> bool:
    """Check if uv is available and probe whether --system is needed."""
    global UV_NEEDS_SYSTEM
    if not shutil.which("uv"):
        return False
    # Probe: try a dry-run install targeting the current Python explicitly.
    # Without --python, uv can ignore the activated venv on some platforms.
    probe = subprocess.run(
        ["uv", "pip", "install", "--dry-run", "--python", sys.executable, "pip"],
        stdout = subprocess.PIPE,
        stderr = subprocess.STDOUT,
    )
    if probe.returncode != 0:
        # Retry with --system (some envs need it when uv can't find a venv)
        probe_sys = subprocess.run(
            ["uv", "pip", "install", "--dry-run", "--system", "pip"],
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
        )
        if probe_sys.returncode != 0:
            return False  # uv is broken, fall back to pip
        UV_NEEDS_SYSTEM = True
    return True


def _filter_requirements(req: Path, skip: set[str]) -> Path:
    """Return a temp copy of a requirements file with certain packages removed."""
    lines = req.read_text(encoding = "utf-8").splitlines(keepends = True)
    filtered = [
        line
        for line in lines
        if not any(line.strip().lower().startswith(pkg) for pkg in skip)
    ]
    tmp = tempfile.NamedTemporaryFile(
        mode = "w",
        suffix = ".txt",
        delete = False,
        encoding = "utf-8",
    )
    tmp.writelines(filtered)
    tmp.close()
    return Path(tmp.name)


def _translate_pip_args_for_uv(args: tuple[str, ...]) -> list[str]:
    """Translate pip flags to their uv equivalents."""
    translated: list[str] = []
    for arg in args:
        if arg == "--no-cache-dir":
            continue  # uv cache is fast; drop this flag
        elif arg == "--force-reinstall":
            translated.append("--reinstall")
        else:
            translated.append(arg)
    return translated


def _build_pip_cmd(args: tuple[str, ...]) -> list[str]:
    """Build a standard pip install command.

    Strips uv-only flags like --upgrade-package that pip doesn't understand.
    """
    cmd = [sys.executable, "-m", "pip", "install"]
    skip_next = False
    for arg in args:
        if skip_next:
            skip_next = False
            continue
        if arg == "--upgrade-package":
            skip_next = True  # skip the flag and its value
            continue
        cmd.append(arg)
    return cmd


def _build_uv_cmd(args: tuple[str, ...]) -> list[str]:
    """Build a uv pip install command with translated flags."""
    cmd = ["uv", "pip", "install"]
    if UV_NEEDS_SYSTEM:
        cmd.append("--system")
    # Always pass --python so uv targets the correct environment.
    # Without this, uv can ignore an activated venv and install into
    # the system Python (observed on Colab and similar environments).
    cmd.extend(["--python", sys.executable])
    cmd.extend(_translate_pip_args_for_uv(args))
    # Torch is pre-installed by install.sh/setup.ps1.  Do not add
    # --torch-backend by default -- it can cause solver dead-ends on
    # CPU-only machines.  Callers that need it can set UV_TORCH_BACKEND.
    _tb = os.environ.get("UV_TORCH_BACKEND", "")
    if _tb:
        cmd.append(f"--torch-backend={_tb}")
    return cmd


def pip_install(
    label: str,
    *args: str,
    req: Path | None = None,
    constrain: bool = True,
) -> None:
    """Build and run a pip install command (uses uv when available, falls back to pip)."""
    constraint_args: list[str] = []
    if constrain and CONSTRAINTS.is_file():
        constraint_args = ["-c", str(CONSTRAINTS)]

    actual_req = req
    temp_reqs: list[Path] = []
    if req is not None and IS_WINDOWS and WINDOWS_SKIP_PACKAGES:
        actual_req = _filter_requirements(req, WINDOWS_SKIP_PACKAGES)
        temp_reqs.append(actual_req)
    if actual_req is not None and NO_TORCH and NO_TORCH_SKIP_PACKAGES:
        actual_req = _filter_requirements(actual_req, NO_TORCH_SKIP_PACKAGES)
        temp_reqs.append(actual_req)
    req_args: list[str] = []
    if actual_req is not None:
        req_args = ["-r", str(actual_req)]

    try:
        if USE_UV:
            uv_cmd = _build_uv_cmd(args) + constraint_args + req_args
            if VERBOSE:
                print(f"   {label}...")
            result = subprocess.run(
                uv_cmd,
                stdout = subprocess.PIPE,
                stderr = subprocess.STDOUT,
            )
            if result.returncode == 0:
                return
            print(_red(f"   uv failed, falling back to pip..."))
            if result.stdout:
                print(result.stdout.decode(errors = "replace"))

        pip_cmd = _build_pip_cmd(args) + constraint_args + req_args
        run(f"{label} (pip)" if USE_UV else label, pip_cmd)
    finally:
        for temp_req in temp_reqs:
            temp_req.unlink(missing_ok = True)


def download_file(url: str, dest: Path) -> None:
    """Download a file using urllib (no curl dependency)."""
    urllib.request.urlretrieve(url, dest)


def patch_package_file(package_name: str, relative_path: str, url: str) -> None:
    """Download a file from url and overwrite a file inside an installed package."""
    result = subprocess.run(
        [sys.executable, "-m", "pip", "show", package_name],
        capture_output = True,
        text = True,
    )
    if result.returncode != 0:
        _step(_LABEL, f"package {package_name} not found, skipping patch", _red)
        return

    location = None
    for line in result.stdout.splitlines():
        if line.lower().startswith("location:"):
            location = line.split(":", 1)[1].strip()
            break

    if not location:
        _step(_LABEL, f"could not locate {package_name}", _red)
        return

    dest = Path(location) / relative_path
    _step(_LABEL, f"patching {dest.name} in {package_name}...", _dim)
    download_file(url, dest)


# -- Main install sequence ---------------------------------------------


def install_python_stack() -> int:
    global USE_UV, _STEP, _TOTAL
    _STEP = 0

    # When called from install.sh (which already installed unsloth into the venv),
    # SKIP_STUDIO_BASE=1 is set to avoid redundant reinstallation of base packages.
    # When called from "unsloth studio update", it is NOT set so base packages
    # (unsloth + unsloth-zoo) are always reinstalled to pick up new versions.
    skip_base = os.environ.get("SKIP_STUDIO_BASE", "0") == "1"
    # When --package is used, install a different package name (e.g. roland-sloth for testing)
    package_name = os.environ.get("STUDIO_PACKAGE_NAME", "unsloth")
    # When --local is used, overlay a local repo checkout after updating deps
    local_repo = os.environ.get("STUDIO_LOCAL_REPO", "")
    base_total = 10 if IS_WINDOWS else 11
    if IS_MACOS:
        base_total -= 1  # triton step is skipped on macOS
    # ROCm torch check step (Linux only, non-macOS, non-no-torch)
    if not IS_WINDOWS and not IS_MACOS and not NO_TORCH:
        base_total += 1
    _TOTAL = (base_total - 1) if skip_base else base_total

    # 1. Try to use uv for faster installs (must happen before pip upgrade
    #    because uv venvs don't include pip by default)
    USE_UV = _bootstrap_uv()

    # 2. Ensure pip is available (uv venvs created by install.sh don't include pip)
    _progress("pip bootstrap")
    if USE_UV:
        run(
            "Bootstrapping pip via uv",
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "pip",
            ],
        )
    else:
        # pip may not exist yet (uv-created venvs omit it). Try ensurepip
        # first, then upgrade. Only fall back to a direct upgrade when pip
        # is already present.
        _has_pip = (
            subprocess.run(
                [sys.executable, "-m", "pip", "--version"],
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL,
            ).returncode
            == 0
        )

        if not _has_pip:
            run(
                "Bootstrapping pip via ensurepip",
                [sys.executable, "-m", "ensurepip", "--upgrade"],
            )
        else:
            run(
                "Upgrading pip",
                [sys.executable, "-m", "pip", "install", "--upgrade", "pip"],
            )

    # 3. Core packages: unsloth-zoo + unsloth (or custom package name)
    if skip_base:
        pass
    elif NO_TORCH:
        # No-torch update path: install unsloth + unsloth-zoo with --no-deps
        # (current PyPI metadata still declares torch as a hard dep), then
        # runtime deps with --no-deps (avoids transitive torch).
        _progress("base packages (no torch)")
        pip_install(
            f"Updating {package_name} + unsloth-zoo (no-torch mode)",
            "--no-cache-dir",
            "--no-deps",
            "--upgrade-package",
            package_name,
            "--upgrade-package",
            "unsloth-zoo",
            package_name,
            "unsloth-zoo",
        )
        pip_install(
            "Installing no-torch runtime deps",
            "--no-cache-dir",
            "--no-deps",
            req = REQ_ROOT / "no-torch-runtime.txt",
        )
        if local_repo:
            pip_install(
                "Overlaying local repo (editable)",
                "--no-cache-dir",
                "--no-deps",
                "-e",
                local_repo,
                constrain = False,
            )
    elif local_repo:
        # Local dev install: update deps from base.txt, then overlay the
        # local checkout as an editable install (--no-deps so torch is
        # never re-resolved).
        _progress("base packages")
        pip_install(
            "Updating base packages",
            "--no-cache-dir",
            "--upgrade-package",
            "unsloth",
            "--upgrade-package",
            "unsloth-zoo",
            req = REQ_ROOT / "base.txt",
        )
        pip_install(
            "Overlaying local repo (editable)",
            "--no-cache-dir",
            "--no-deps",
            "-e",
            local_repo,
            constrain = False,
        )
    elif package_name != "unsloth":
        # Custom package name (e.g. roland-sloth for testing) — install directly
        _progress("base packages")
        pip_install(
            f"Installing {package_name}",
            "--no-cache-dir",
            package_name,
        )
    else:
        # Update path: upgrade only unsloth + unsloth-zoo while preserving
        # existing torch/CUDA installations.  Torch is pre-installed by
        # install.sh / setup.ps1; --upgrade-package targets only base pkgs.
        _progress("base packages")
        pip_install(
            "Updating base packages",
            "--no-cache-dir",
            "--upgrade-package",
            "unsloth",
            "--upgrade-package",
            "unsloth-zoo",
            req = REQ_ROOT / "base.txt",
        )

    # 2b. AMD ROCm: reinstall torch with HIP wheels if the host has ROCm but the
    #     venv received CPU-only torch (common when pip resolves torch from PyPI).
    #     Must come immediately after base packages so torch is present for inspection.
    if not IS_WINDOWS and not IS_MACOS and not NO_TORCH:
        _progress("ROCm torch check")
        _ensure_rocm_torch()

    # Windows + AMD GPU: PyTorch does not publish ROCm wheels for Windows.
    # Detect and warn so users know manual steps are needed for GPU training.
    if IS_WINDOWS and not NO_TORCH and not _has_usable_nvidia_gpu():
        # Validate actual AMD GPU presence (not just tool existence)
        import re as _re_win

        def _win_amd_smi_has_gpu(stdout: str) -> bool:
            return bool(_re_win.search(r"(?im)^gpu\s*[:\[]\s*\d", stdout))

        _win_amd_gpu = False
        for _wcmd, _check_fn in (
            (["hipinfo"], lambda out: "gcnarchname" in out.lower()),
            (["amd-smi", "list"], _win_amd_smi_has_gpu),
        ):
            _wexe = shutil.which(_wcmd[0])
            if not _wexe:
                continue
            try:
                _wr = subprocess.run(
                    [_wexe, *_wcmd[1:]],
                    stdout = subprocess.PIPE,
                    stderr = subprocess.DEVNULL,
                    text = True,
                    timeout = 10,
                )
            except Exception:
                continue
            if _wr.returncode == 0 and _check_fn(_wr.stdout):
                _win_amd_gpu = True
                break
        if _win_amd_gpu:
            _safe_print(
                _dim("  Note:"),
                "AMD GPU detected on Windows. ROCm-enabled PyTorch must be",
            )
            _safe_print(
                " " * 8,
                "installed manually. See: https://docs.unsloth.ai/get-started/install-and-update/amd",
            )

    # 3. Extra dependencies
    _progress("unsloth extras")
    pip_install(
        "Installing additional unsloth dependencies",
        "--no-cache-dir",
        req = REQ_ROOT / "extras.txt",
    )

    # 3b. Extra dependencies (no-deps) -- audio model support etc.
    _progress("extra codecs")
    pip_install(
        "Installing extras (no-deps)",
        "--no-deps",
        "--no-cache-dir",
        req = REQ_ROOT / "extras-no-deps.txt",
    )

    # 4. Overrides (torchao, transformers) -- force-reinstall
    #    Skip entirely when torch is unavailable (e.g. Intel Mac GGUF-only mode)
    #    because overrides.txt contains torchao which requires torch.
    if NO_TORCH:
        _progress("dependency overrides (skipped, no torch)")
    else:
        _progress("dependency overrides")
        pip_install(
            "Installing dependency overrides",
            "--force-reinstall",
            "--no-cache-dir",
            req = REQ_ROOT / "overrides.txt",
        )

    # 5. Triton kernels (no-deps, from source)
    #    Skip on Windows (no support) and macOS (no support).
    if not IS_WINDOWS and not IS_MACOS:
        _progress("triton kernels")
        pip_install(
            "Installing triton kernels",
            "--no-deps",
            "--no-cache-dir",
            req = REQ_ROOT / "triton-kernels.txt",
            constrain = False,
        )

    # # 6. Patch: override llama_cpp.py with fix from unsloth-zoo  feature/llama-cpp-windows-support branch
    # patch_package_file(
    #     "unsloth-zoo",
    #     os.path.join("unsloth_zoo", "llama_cpp.py"),
    #     "https://raw.githubusercontent.com/unslothai/unsloth-zoo/refs/heads/main/unsloth_zoo/llama_cpp.py",
    # )

    # # 7a. Patch: override vision.py with fix from unsloth PR #4091
    # patch_package_file(
    #     "unsloth",
    #     os.path.join("unsloth", "models", "vision.py"),
    #     "https://raw.githubusercontent.com/unslothai/unsloth/80e0108a684c882965a02a8ed851e3473c1145ab/unsloth/models/vision.py",
    # )

    # # 7b. Patch : override save.py with fix from feature/llama-cpp-windows-support
    # patch_package_file(
    #     "unsloth",
    #     os.path.join("unsloth", "save.py"),
    #     "https://raw.githubusercontent.com/unslothai/unsloth/refs/heads/main/unsloth/save.py",
    # )

    # 8. Studio dependencies
    _progress("studio deps")
    pip_install(
        "Installing studio dependencies",
        "--no-cache-dir",
        req = REQ_ROOT / "studio.txt",
    )

    # 9. Data-designer dependencies
    _progress("data designer deps")
    pip_install(
        "Installing data-designer base dependencies",
        "--no-cache-dir",
        req = SINGLE_ENV / "data-designer-deps.txt",
    )

    # 10. Data-designer packages (no-deps to avoid conflicts)
    _progress("data designer")
    pip_install(
        "Installing data-designer",
        "--no-cache-dir",
        "--no-deps",
        req = SINGLE_ENV / "data-designer.txt",
    )

    # 11. Local Data Designer seed plugin
    if not LOCAL_DD_UNSTRUCTURED_PLUGIN.is_dir():
        _safe_print(
            _red(
                f"❌ Missing local plugin directory: {LOCAL_DD_UNSTRUCTURED_PLUGIN}",
            ),
        )
        return 1
    _progress("local plugin")
    pip_install(
        "Installing local data-designer unstructured plugin",
        "--no-cache-dir",
        "--no-deps",
        str(LOCAL_DD_UNSTRUCTURED_PLUGIN),
        constrain = False,
    )

    # 12. Patch metadata for single-env compatibility
    _progress("finalizing")
    run(
        "Patching single-env metadata",
        [sys.executable, str(SINGLE_ENV / "patch_metadata.py")],
    )

    # 13. Final check (silent; third-party conflicts are expected)
    subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        stdout = subprocess.DEVNULL,
        stderr = subprocess.DEVNULL,
    )

    _step(_LABEL, "installed")
    return 0


if __name__ == "__main__":
    sys.exit(install_python_stack())
