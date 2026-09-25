# CI performance and preserved checks

The required `ci (3.13)` job still runs all four tox environments: installed-package
tests, Ruff/format, the Mypy/Pyright/Pylint ratchet, and build/Twine. Two environments
run concurrently and any failure fails the required job. Analyzer commands,
versions, normalization and the immutable debt ceiling are unchanged.

CPU runners constrain Torch to `2.14.0+cpu` before tox installs package dependencies.
This retains the measured baseline's Torch API version. The constraint and CPU
index are CI environment settings, not changes to project/runtime dependencies.
Pip caching is enabled. A separate `pip-cpu-v1` namespace avoids setup-python's
fallback restore of obsolete multi-gigabyte CUDA downloads. Normal package builds
and installed-distribution validation are retained.

The 79 selected tests (78 passes and one expected CUDA skip) are unchanged, as are
all assertions. The slowest baseline test was 0.461 seconds; the existing numerical
and gradient fixtures already use small real models. Reducing them is unnecessary.
Every covered production line and branch is compared with the baseline, along with
exact normalized analyzer diagnostic multisets. No runtime source is changed.

The former runner-wide disk deletion is removed after CPU dependencies reduced
`.tox` from about12GB to2.3GB. CI records cache and disk usage on every execution;
full runs without cleanup retained about84GB of free disk during the audit.
`scripts/ci_test_metrics.py` observes the unchanged pytest selection, records
identities/durations and returns pytest's actual status. JSON/JUnit/branch coverage
and static reports are uploaded as `ci-validation-evidence` even after failures.

The release workflow is unchanged and still validates the complete tagged checkout,
builds and checks distributions, and uses trusted publishing. No "latest main"
evidence substitutes for tagged-checkout validation. The small existing test-env
pip precommands are retained (under a second in measured warm runs); their removal
would have negligible impact compared with dependency resolution and installation.
