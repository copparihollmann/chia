"""Content-addressed cache tags for :mod:`chia.base.cache`.

The cache is keyed by ``_chia_tag``, a string the call site supplies::

    run_verilator_test.chia_remote(design, _chia_tag=f"iter{i}_opt{j}")

``_tag_filename`` hashes that string, so the key covers *what the caller chose to name the call* and
nothing else. For a loop index that is exactly right. For an expensive build, simulation or synthesis
it is a correctness bug waiting to happen: edit the RTL, keep the tag, and the cache serves the
previous design's result. Nothing errors, and the number is wrong in a way no downstream check can
see — a synthesis table can report a prior candidate's area under the current candidate's name.

This module builds the tag the other way round: from the *content* of everything the call depends on.
Same inputs, same tag, cache hit. One byte different anywhere, different tag, the work re-runs.

::

    from chia.base.content_tag import content_tag

    tag = content_tag(
        "synth_mxu0",
        files=[sv_path],
        dirs=[liberty_dir],
        params={"clock_ns": 1.0, "flatten": "none"},
        tools={"yosys": yosys_version},
    )
    run_synthesis.chia_remote(sv_path, _chia_tag=tag)

The cache itself is unchanged and needs no changes: it was always correct given a correct key.

Design choices worth knowing:

* **A missing input raises.** Silently skipping one would make the "input absent" and "input present"
  cases hash the same, which is the failure this module exists to prevent, one level up.
* **Paths are hashed alongside contents.** Renaming a file changes the tag even if the bytes do not,
  because a build that reads by name is not the same build.
* **The readable prefix is kept.** ``_tag_filename`` slugs the tag into the on-disk filename, so
  ``synth_mxu0@1a2b…`` stays greppable in a cache directory, unlike a bare digest. It truncates that
  slug at 80 characters, so keep ``name`` under ~63 chars if you want the content digest to remain
  visible in the filename — beyond that the entry is still unique (``_tag_filename`` appends its own
  hash of the full tag) but you can no longer tell from ``ls`` which content it addressed.
* **No Ray import.** Pure stdlib, so it can be used to build a tag before a cluster exists.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

#: Directory names never walked when hashing a tree. Build scratch and VCS metadata change constantly
#: without changing what a build reads, so including them would defeat the cache entirely.
DEFAULT_EXCLUDE = frozenset({
    ".git", ".hg", ".svn", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache",
    ".venv", "node_modules", ".DS_Store",
})

_CHUNK = 1 << 20  # 1 MiB
_DIGEST_LEN = 16  # hex chars kept in the tag; 64 bits of collision resistance is ample for a cache key


class MissingCacheInput(FileNotFoundError):
    """An input named in a content tag does not exist.

    Deliberately fatal. Skipping it would produce the same tag whether or not the input was there,
    so a run with a missing dependency would silently collide with one that had it.
    """


def sha256_file(path: str | os.PathLike[str]) -> str:
    """Streaming sha256 of one file's bytes. Symlinks are followed."""
    p = Path(path)
    if not p.is_file():
        raise MissingCacheInput(f"not a file: {p}")
    h = hashlib.sha256()
    with p.open("rb") as fh:
        while chunk := fh.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def sha256_dir(path: str | os.PathLike[str], exclude: Iterable[str] = DEFAULT_EXCLUDE) -> str:
    """Deterministic sha256 over a directory tree: sorted relative paths, each with its content.

    Both the path and the bytes go in, so a rename is a change. Traversal order is sorted rather than
    ``os.walk`` order, which is filesystem-dependent and would make the same tree hash differently on
    two machines — turning a shared cache into a per-machine one without anyone noticing.
    """
    root = Path(path)
    if not root.is_dir():
        raise MissingCacheInput(f"not a directory: {root}")
    skip = set(exclude)
    h = hashlib.sha256()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in skip)
        for name in sorted(filenames):
            if name in skip:
                continue
            f = Path(dirpath) / name
            rel = f.relative_to(root).as_posix()
            h.update(rel.encode("utf-8"))
            h.update(b"\0")
            try:
                h.update(bytes.fromhex(sha256_file(f)))
            except MissingCacheInput:
                # A dangling symlink: record its absence explicitly rather than skipping, so a tree
                # with a broken link does not hash the same as one without the link at all.
                h.update(b"<dangling>")
            h.update(b"\0")
    return h.hexdigest()


def _canonical(value: Any) -> str:
    """Stable text for a parameter value. Sorted keys, no incidental whitespace."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def content_tag(
    name: str,
    *,
    files: Iterable[str | os.PathLike[str]] = (),
    dirs: Iterable[str | os.PathLike[str]] = (),
    params: Mapping[str, Any] | None = None,
    tools: Mapping[str, Any] | None = None,
    extra: str = "",
    exclude: Iterable[str] = DEFAULT_EXCLUDE,
    digest_len: int = _DIGEST_LEN,
) -> str:
    """Build a ``_chia_tag`` that covers every input by content.

    ``name``    readable prefix, kept in the tag and in the on-disk cache filename.
    ``files``   individual input files (RTL, constraints, a test binary).
    ``dirs``    input trees (a liberty library, a source tree).
    ``params``  the call's own arguments — anything that changes the output.
    ``tools``   tool name -> version. A yosys upgrade must invalidate a synthesis result; without
                this it does not, and the cache quietly mixes results from two toolchains.
    ``extra``   free-form discriminator for anything the above cannot express.

    Returns ``"<name>@<digest>"``. Raises :class:`MissingCacheInput` if any named input is absent.
    """
    h = hashlib.sha256()
    h.update(b"chia-content-tag-v1\0")
    h.update(name.encode("utf-8"))
    h.update(b"\0")

    # Sorted so the caller's argument order cannot change the tag: two call sites listing the same
    # two files in different orders describe the same work and must hit the same entry.
    for f in sorted(str(Path(x)) for x in files):
        h.update(b"file\0")
        h.update(f.encode("utf-8"))
        h.update(b"\0")
        h.update(bytes.fromhex(sha256_file(f)))
        h.update(b"\0")

    for d in sorted(str(Path(x)) for x in dirs):
        h.update(b"dir\0")
        h.update(d.encode("utf-8"))
        h.update(b"\0")
        h.update(bytes.fromhex(sha256_dir(d, exclude=exclude)))
        h.update(b"\0")

    if params:
        h.update(b"params\0")
        h.update(_canonical(dict(params)).encode("utf-8"))
        h.update(b"\0")

    if tools:
        h.update(b"tools\0")
        h.update(_canonical(dict(tools)).encode("utf-8"))
        h.update(b"\0")

    if extra:
        h.update(b"extra\0")
        h.update(extra.encode("utf-8"))
        h.update(b"\0")

    return f"{name}@{h.hexdigest()[:digest_len]}"


def explain(
    name: str,
    *,
    files: Iterable[str | os.PathLike[str]] = (),
    dirs: Iterable[str | os.PathLike[str]] = (),
    params: Mapping[str, Any] | None = None,
    tools: Mapping[str, Any] | None = None,
    extra: str = "",
    exclude: Iterable[str] = DEFAULT_EXCLUDE,
) -> dict:
    """The per-input digests behind a tag, for debugging a cache miss nobody expected.

    Answering "why did this re-run?" by re-deriving the tag by hand is tedious enough that people
    stop asking, and a cache whose misses are unexplained gets disabled.
    """
    return {
        "tag": content_tag(name, files=files, dirs=dirs, params=params, tools=tools,
                           extra=extra, exclude=exclude),
        "name": name,
        "files": {str(Path(f)): sha256_file(f) for f in sorted(str(Path(x)) for x in files)},
        "dirs": {str(Path(d)): sha256_dir(d, exclude=exclude)
                 for d in sorted(str(Path(x)) for x in dirs)},
        "params": _canonical(dict(params)) if params else None,
        "tools": _canonical(dict(tools)) if tools else None,
        "extra": extra or None,
    }
