from swebench.harness.test_spec.test_spec import make_test_spec, TestSpec
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER
try:
    from swebench.harness.log_parsers import get_eval_type  # swebench<4
except ImportError:
    # swebench>=4 removed this helper; reimplement locally to avoid pinning.
    from swebench.harness.constants import EvalType, FAIL_ONLY_REPOS

    def get_eval_type(test_spec: TestSpec) -> str:
        if test_spec.repo in FAIL_ONLY_REPOS:
            return EvalType.FAIL_ONLY
        return EvalType.PASS_AND_FAIL
from swebench.harness.grading import get_eval_tests_report, get_resolution_status

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    END_TEST_OUTPUT,
    FAIL_TO_FAIL,
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
    KEY_PREDICTION,
    MAP_REPO_VERSION_TO_SPECS,
    PASS_TO_FAIL,
    PASS_TO_PASS,
    RESET_FAILED,
    START_TEST_OUTPUT,
    TESTS_ERROR,
    TESTS_TIMEOUT,
    EvalType,
    ResolvedStatus,
    TestStatus,
)

import re


def harden_eval_script_for_overlay(eval_script: str) -> str:
    """Make a generated SWE-bench eval script resilient to a rootless-overlay race.

    Under rootless overlay storage, libtool's relink of a bundled C shared library (e.g. jq's ``oniguruma``
    submodule) fails with ``ln: failed to create symbolic link 'libonig.so.5': File exists``. The pre-built ``.so``
    symlink lives only in the image's lower layer; after ``make clean`` whiteouts it, the parallel ``make -j`` relink
    races with the overlay whiteout and ``ln -s`` spuriously sees the masked entry as still present, aborting the build
    (and therefore scoring the gold patch 0).

    A plain serial ``make`` retry completes the relink because the rebuilt artifacts have since been copied up into the
    overlay upper layer. We therefore turn each standalone parallel/bare build invocation (``make`` / ``make -j...``)
    into ``make ... || make ... || make``. This is a no-op for healthy builds and never masks a genuine failure: a
    truly broken build still fails on the serial retry, leaving the reward at 0. Only the top-level build step is
    rewritten; ``make clean``, ``make check``, ``make install`` and other targets are left untouched.
    """
    return re.sub(
        r"(?m)^(?P<indent>[ \t]*)make(?P<jobs>[ \t]+-j\S*)?[ \t]*$",
        lambda m: (
            f"{m.group('indent')}make{m.group('jobs') or ''} "
            f"|| make{m.group('jobs') or ''} || make"
        ),
        eval_script,
    )


def get_logs_eval(test_spec: TestSpec, content: str) -> tuple[dict[str, str], bool]:
    """
    Retrieve evaluation results for a task instance from its corresponding log file

    Args:
        log_fp (str): path to log file
    Returns:
        bool: whether the patch applied successfully
        dict: status map

    modified from swebench/harness/grading.py
    """
    repo = test_spec.repo
    version = test_spec.version
    log_parser = MAP_REPO_TO_PARSER[repo]
    test_cmd = MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"]
    if isinstance(test_cmd, list):
        test_cmd = test_cmd[-1]

    # with open(log_fp) as f:
    # # TODO fix constant here
    bad_codes = list(
        filter(
            lambda x: x in content,
            [
                APPLY_PATCH_FAIL,
                RESET_FAILED,
                TESTS_ERROR,
                TESTS_TIMEOUT,
            ],
        )
    )
    if bad_codes:
        return {}, False

    # Get status map of evaluation results. Prefer the official Start/End markers emitted by the
    # eval script: the legacy ``content.split(test_cmd)[-1]`` keeps only the output *after* the last
    # per-test command, which silently drops every other test when ``test_cmd`` is a list (e.g.
    # SWE-bench Multilingual runs one `mvnd test -Dtest=...` / `cargo test` / `go test` per test).
    # Fall back to the legacy split only when the markers are absent.
    if START_TEST_OUTPUT in content and END_TEST_OUTPUT in content:
        test_content = content.split(START_TEST_OUTPUT)[1].split(END_TEST_OUTPUT)[0]
    else:
        test_content = content.split(test_cmd)[-1]
    status_map = log_parser(test_content, test_spec)
    if not status_map:
        # Some scripts emit results outside the markers (e.g. to stderr); retry on the full log.
        status_map = log_parser(content, test_spec)
    return status_map, True


def swebench_report(ds, test_output):
    test_spec = make_test_spec(ds)
    eval_status_map, found = get_logs_eval(test_spec, test_output)
    eval_ref = {
        KEY_INSTANCE_ID: test_spec.instance_id,
        FAIL_TO_PASS: test_spec.FAIL_TO_PASS,
        PASS_TO_PASS: test_spec.PASS_TO_PASS,
    }
    report = get_eval_tests_report(
        eval_status_map, eval_ref, eval_type=get_eval_type(test_spec)
    )
    return report


def swebench_parse(ds, test_output):
    test_spec = make_test_spec(ds)
    eval_status_map, found = get_logs_eval(test_spec, test_output)
    return eval_status_map
