# Visible static-analysis debt

Coral 0.1.15 retains three Pylint diagnostics already present at `dc1c1f7`:
`R0914` (relation-mask local count), `W0221` (multi-input forward signature), and
`R0801` (architecture summary duplication). Mypy and Pyright report zero errors.
These identities were compared against the original release source; they are
not regressions introduced by relation scaling or the additive-mask correction.

The default `tox` gate runs strict tests, Ruff/format, package builds, and
`scripts/static_analysis_ratchet.py`. Every current diagnostic remains printed.
A ratchet pass means **no new static-analysis debt**, not zero Pylint messages.
The strict developer commands `tox -e lint,typecheck` remain available.

`diagnostics.json` records tool versions, analyzer-configuration hash, lexical
scope, source text, rule and message. Duplicate occurrences are counted. Line
shifts alone do not create new identities; an additional occurrence or changed
scope does. Invalid reports, failed analyzers and version/policy mismatches fail
closed. Machine-readable raw output and a count summary are written under the
tox environment's temporary directory.

Baseline growth is checked against its first committed version and, in PR CI,
the target branch's baseline. Full Git history is required. CI never records or
updates the baseline. When debt disappears, remove those exact entries in a
reviewed change; the ratchet reports reductions without silently editing files.
The shared checker contract is covered by deterministic unit tests for new debt,
duplicate counts, line movement, malformed reports, and baseline-growth attempts.

Release scope: relation bias defaults to the old scale of 1.0; new explicit scales
persist through architecture parsing without changing weight keys. The additive
inference correction uses the existing encoder parameters and is checked against
the standard unfused training computation. No type-specific, D4, width-screen or
other experiment infrastructure is included.
