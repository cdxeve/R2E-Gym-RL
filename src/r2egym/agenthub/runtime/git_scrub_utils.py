"""Scrub a task repo's git history so the agent can't read the solution from it.

SWE-Bench-Verified/Pro images are built by ``git clone`` + ``git reset --hard <base_commit>``,
which leaves the whole upstream history — including the fix commit for the issue under test —
reachable via the default branch, the clone reflog, or (Pro) the ``origin`` remote. An agent
could ``git checkout <fix_commit> -- <files>`` to grab the gold patch without solving anything.

The leak is any ref that reaches a commit *outside* ``base_commit``'s own history: the future fix
commit, ``origin``'s real tip, and release tags cut *after* the base. We delete exactly those — every
remote (config + remote-tracking refs, which also disables ``git fetch``) and every branch/tag that
is **not** an ancestor of ``base_commit`` — plus the reflog. Ancestor tags are deliberately KEPT:
they predate ``base_commit`` (so cannot reveal the fix) and ``setuptools_scm``-style versioning reads
the package version from the nearest ancestor tag via ``git describe`` when the repo's ``install``
step re-runs at eval time. Dropping *every* tag made that fall back to ``0.1.dev…``, tripping pytest's
own ``minversion`` guard so no test ran and the gold-patch reward was 0. Keeping ancestor tags makes
version detection identical to the pre-scrub image, so no other repo's outcome changes.

We do **not** run ``git gc``/``git prune`` — repacking a large history at setup is slow, and the
reward path still needs the (now unreachable) fix-commit object for an explicit-hash ``git checkout``.
Objects stay; only the leaking refs/remotes/reflog are gone, so the repo looks normal
(``git log``/``branch``/``remote``/``fetch`` reveal nothing past the base; ``git tag`` shows only past
releases). Deletion is per-ref via ``git update-ref -d`` (rewrites ``packed-refs`` in place, so the
ancestor tags packed alongside survive — a blanket ``rm packed-refs`` would drop them). HEAD already
sits on ``base_commit`` and a branch is an ancestor of itself, so the current branch is kept as-is.
"""

from typing import Callable, Optional

# POSIX-sh script run once in the repo; every step is guarded so a non-git dir is a clean no-op.
_SCRUB_SCRIPT = r"""
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || exit 0
GD=$(git rev-parse --git-dir 2>/dev/null) || exit 0
# Commit the image was reset --hard onto (current HEAD). Refs that can reach a commit NOT contained
# in BASE's history are the leak; refs that are ancestors of BASE are safe (and some are needed for
# `git describe`-based versioning), so they are kept.
BASE=$(git rev-parse --verify -q HEAD) || exit 0
# Remove every remote (config entry + refs/remotes), disabling origin/* inspection and `git fetch`.
for r in $(git remote 2>/dev/null); do git remote remove "$r" >/dev/null 2>&1 || true; done
# Precompute commit generation numbers so the per-ref ancestry test below is O(1) instead of a full
# history walk each time (tens of seconds -> a few seconds on large repos such as django). Best-effort
# cache only; the loop stays correct without it.
git commit-graph write --reachable >/dev/null 2>&1 || true
# Delete every ref (any namespace: branches, tags, remote-tracking, plus refs/pull, refs/notes,
# refs/replace, refs/stash, ...) that is NOT an ancestor of BASE -- i.e. that could reach the future
# fix commit. `git update-ref -d` rewrites packed-refs in place, so ancestor tags packed in the same
# file survive -- do NOT `rm packed-refs`, which would drop them too. HEAD is a symref and is not
# listed by for-each-ref, so it is never touched.
git for-each-ref --format='%(refname)' 2>/dev/null \
  | while read ref; do
      git merge-base --is-ancestor "$ref" "$BASE" 2>/dev/null && continue
      git update-ref -d "$ref" >/dev/null 2>&1 || true
    done
# Drop the reflog: its entries still record the pre-reset tip, which reaches the fix commit.
rm -rf "$GD/logs"
git reflog expire --expire=now --all >/dev/null 2>&1 || true
exit 0
"""


def scrub_git_history(
    run: Callable[..., tuple],
    logger=None,
    timeout: Optional[int] = 300,
) -> None:
    """Remove refs/remotes/reflog that point past ``base_commit`` in the task repo.

    Best-effort hardening: ``run`` is the runtime's command executor; failures are swallowed/logged
    and must never abort env setup.
    """
    run_kwargs = {} if timeout is None else {"timeout": timeout}
    try:
        run(_SCRUB_SCRIPT, **run_kwargs)
    except Exception as e:  # pragma: no cover - defensive
        if logger is not None:
            logger.error(f"git history scrub failed: {e!r}")
