"""Shared recovery for a corrupt ``.git/index`` during patch extraction.

``get_patch`` (``git add -A && git diff --cached``) is the single point where a
rollout's edits are exported as a diff. On some SWE-bench tasks (observed on
matplotlib) a stray PNG overwrote ``.git/index`` mid-rollout, so git aborted
with ``error: bad signature 0x474e5089 / fatal: index file corrupt`` instead of
emitting a diff. A stale ``index.lock`` produces a similar dead end. In every
such case the working tree and git objects are intact, so the index can be
rebuilt from HEAD without losing the agent's edits.

This module centralizes the detection and recovery for the Docker runtime.
"""

from typing import Callable, Optional

# Markers git emits when ``.git/index`` is unusable (corrupt index or stale
# ``index.lock``).
_INDEX_FAILURE_MARKERS = (
    "index file corrupt",
    "index file smaller than expected",
    "bad signature 0x",
    "cannot read the index",
    "unable to read index",
    "broken index",
    "index.lock",
)


def looks_like_index_failure(output: str) -> bool:
    """Return True when ``output`` is a git index error rather than a diff.

    A real diff always carries a ``diff --git`` header; a bare git index error
    never does. Requiring the header to be absent avoids misclassifying a valid
    patch (which may legitimately contain these marker strings in file content)
    as a failure.
    """
    if not isinstance(output, str) or "diff --git " in output:
        return False
    low = output.lower()
    return any(marker in low for marker in _INDEX_FAILURE_MARKERS)


def get_patch_with_index_recovery(
    run: Callable[..., tuple],
    logger,
    timeout: Optional[int] = None,
) -> str:
    """Run ``git add -A && git diff --cached``, recovering from a bad index.

    ``run`` is the runtime's command executor returning a ``(output, ...)``
    tuple. ``timeout`` is forwarded to ``run`` when provided (some backends
    require an explicit timeout).
    """
    run_kwargs = {} if timeout is None else {"timeout": timeout}

    output, _ = run("git add -A && git diff --cached", **run_kwargs)
    if looks_like_index_failure(output):
        logger.warning(
            f"get_patch: .git/index unusable ({output.strip()!r}); "
            "rebuilding it from HEAD and re-extracting the diff"
        )
        run("rm -f .git/index .git/index.lock && git read-tree HEAD", **run_kwargs)
        output, _ = run("git add -A && git diff --cached", **run_kwargs)
    return output
