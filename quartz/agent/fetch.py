"""Where the weights live, and how they get there.

The whole model is one file either way it ships, so fetching is a download, a
digest check and a rename. There is no shard index, no resume protocol and no
client library, because those exist to manage two hundred files and we have one.

Two artifacts, for two runtimes. The `.pkl` checkpoint is what Python loads: a
named parameter tree, the geometry, and nothing else. The `.ingot` is the device
container, read positionally by a runtime that already knows the module tree,
and it is listed here so it can be fetched, not because `Quartz` can load it.

Three ways to point at weights, in the order they are consulted:

1. an explicit path, passed to `Quartz(weights=...)` or `fetch_weights(url=...)`
2. `QUARTZ_WEIGHTS`, an absolute path to a file, which is how an air-gapped or
   CI machine says "it is already here, do not go to the network"
3. the cache under `QUARTZ_HOME`, defaulting to `~/.cache/quartz`
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import urllib.error
import urllib.request
import warnings
from pathlib import Path
from urllib.parse import urlsplit

__all__ = ["DEFAULT_RELEASE", "RELEASES", "fetch_weights", "weights_path"]

#: Published artifacts, name to URL. The local filename comes from the URL, so
#: the registry is the only place a release is described. The digest is not
#: pinned here but fetched from a `.sha256` sidecar beside the file, because a
#: re-cut release cannot be described by a constant that shipped before it.
_RELEASE_BASE = "https://github.com/FareedKhan-dev/quartz/releases/download"
RELEASES: dict[str, str] = {
    "quartz-base": f"{_RELEASE_BASE}/v0.1.0/quartz-base.pkl",
    "quartz-base-heads": f"{_RELEASE_BASE}/v0.1.0/quartz-base-heads.pkl",
    "quartz-base-ingot": f"{_RELEASE_BASE}/v0.1.0/quartz.ingot",
}
DEFAULT_RELEASE = "quartz-base"

_CHUNK = 1 << 20
_TIMEOUT = 60


def weights_path(name: str = DEFAULT_RELEASE, cache_dir: str | os.PathLike | None = None
                 ) -> Path:
    """Where release `name` sits on this machine, whether or not it is there yet.

    Pure path arithmetic: nothing is created and nothing is downloaded, so a
    caller can print the location, check it, or decide to point somewhere else.
    `QUARTZ_WEIGHTS` overrides everything, since a machine that has been handed
    a file should never be asked to go and find another one.
    """
    override = os.environ.get("QUARTZ_WEIGHTS", "").strip()
    if override and cache_dir is None:
        return Path(override).expanduser()
    root = Path(cache_dir).expanduser() if cache_dir else _cache_home()
    url = RELEASES.get(name)
    if url:
        return root / Path(urlsplit(url).path).name
    return root / (name if Path(name).suffix else f"{name}.pkl")


def fetch_weights(name: str = DEFAULT_RELEASE, *, url: str | None = None,
                  sha256: str | None = None,
                  cache_dir: str | os.PathLike | None = None,
                  force: bool = False, progress: bool | None = None) -> Path:
    """Return a local path to the weights, downloading them once if needed.

    Downloads into a `.part` file next to the target and renames on success, so
    an interrupted fetch can never leave a half-written file that loads, reads
    a truncated directory, and fails somewhere far less obvious.

    Args:
        name: a key in RELEASES, or a filename to place in the cache.
        url: fetch this instead of the registered URL.
        sha256: expected digest. When None the `.sha256` sidecar beside the URL
            is tried, and a file with no digest available is warned about rather
            than refused, because an unverifiable file is still better than no
            model on a machine with no sidecar published.
        cache_dir: overrides QUARTZ_HOME for this call.
        force: re-download even when the file is already present.
        progress: print a percentage to stderr. Defaults to on when stderr is a
            terminal, off when it is a log.

    Raises:
        RuntimeError: the download failed, or the digest did not match.
    """
    target = weights_path(name, cache_dir)
    if target.exists() and not force:
        if target.stat().st_size == 0:
            raise RuntimeError(
                f"{target} exists but is empty; delete it, or pass force=True")
        return target

    source = url or RELEASES.get(name)
    if source is None:
        raise RuntimeError(
            f"no download URL for {name!r} (known: {sorted(RELEASES)}). Pass "
            f"url=, or set QUARTZ_WEIGHTS to a local checkpoint.")

    target.parent.mkdir(parents=True, exist_ok=True)
    if progress is None:
        progress = sys.stderr.isatty()
    if sha256 is None:
        sha256 = _sidecar_digest(source)
        if sha256 is None:
            warnings.warn(
                f"no {source}.sha256 published; {target.name} will be used "
                f"unverified", RuntimeWarning, stacklevel=2)

    part = target.with_suffix(target.suffix + ".part")
    digest = _download(source, part, progress=progress)
    if sha256 and digest != sha256.lower():
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"digest mismatch for {source}: expected {sha256.lower()}, got {digest}")
    os.replace(part, target)
    return target


def _download(url: str, part: Path, *, progress: bool) -> str:
    """Stream `url` into `part`, returning the hex sha256 of what arrived."""
    request = urllib.request.Request(url, headers={"User-Agent": "quartz"})
    sha = hashlib.sha256()
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
            total = int(response.headers.get("Content-Length") or 0)
            seen = 0
            with open(part, "wb") as out:
                while chunk := response.read(_CHUNK):
                    out.write(chunk)
                    sha.update(chunk)
                    seen += len(chunk)
                    if progress:
                        _tick(seen, total)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"could not download {url}: {exc}. Download it by hand and set "
            f"QUARTZ_WEIGHTS to the file, or pass weights= to Quartz.") from exc
    if progress:
        print(file=sys.stderr)
    return sha.hexdigest()


def _sidecar_digest(url: str) -> str | None:
    """The first token of `<url>.sha256`, or None when nothing is published."""
    request = urllib.request.Request(f"{url}.sha256",
                                     headers={"User-Agent": "quartz"})
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
            text = response.read(256).decode("utf-8", "replace")
    except (urllib.error.URLError, OSError, TimeoutError):
        return None
    token = text.split()[0].strip().lower() if text.split() else ""
    return token if len(token) == 64 else None


def _tick(seen: int, total: int) -> None:
    """One in-place progress line. Bytes when the server declines to say how
    many are coming, which happens more often than the documentation admits."""
    width = shutil.get_terminal_size((80, 20)).columns
    if total:
        line = f"  quartz: {seen / 1e6:6.2f} / {total / 1e6:.2f} MB"
    else:
        line = f"  quartz: {seen / 1e6:6.2f} MB"
    print(line[: width - 1].ljust(width - 1), end="\r", file=sys.stderr, flush=True)


def _cache_home() -> Path:
    """`QUARTZ_HOME`, else the platform cache directory, else `~/.cache`."""
    home = os.environ.get("QUARTZ_HOME", "").strip()
    if home:
        return Path(home).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or "~/AppData/Local"
        return Path(base).expanduser() / "quartz" / "cache"
    if sys.platform == "darwin":
        return Path("~/Library/Caches/quartz").expanduser()
    base = os.environ.get("XDG_CACHE_HOME") or "~/.cache"
    return Path(base).expanduser() / "quartz"
