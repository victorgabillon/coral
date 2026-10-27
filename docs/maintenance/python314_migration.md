Python 3.14 and PyTorch migration, validated 27 October 2026

The ordinary checkout uses `migrate/python314`, based on public main
`74b61d49521b963297622e1004516ddb681fcf9e`. Validated code/configuration commit:
`7ec9b590a4d604cfe8b42a600f82ab02599ef048`. No worktree, publication, push or
merge was created. CPython 3.14.10 uses the GIL and disables JIT execution.

The selected stable PyTorch artifact is **2.14.1+cpu**, CPython 3.14 Linux wheel
SHA-256 `35103e180214793207f95d7e12e100ecba3ace6f5fb37e1fcefbb07e096ed180`.
No 2.15 release candidate was used. Its installed source and extension files
were checked against the official wheel. CUDA and ROCm are unavailable in this
isolated environment; existing GPU environments were untouched.

Safe changes include native bounded converter/evaluator generics, a Tensor
result default with explicit support for TransformerOne's existing indexed
lists, runtime-resolvable public annotations, removal of the whole-file mypy
suppression in TransformerOne, explicit weights-only direct-path loading with
existing device routing, scalar evaluator inference mode, standard Module
serialization that omits transient compiler state, and simpler relation-mask
indexing with identical accumulation order. No architecture or execution
backend was added. Production lines decrease from 2,170 to 2,164; Any mentions
fall from 17 to 12, and type-ignore directives from two to one.

Full tox passes: **84 tests pass, one existing CUDA-device test skips**; Ruff
check/format, strict mypy and Pyright, Pylint 10/10, wheel/sdist and strict Twine
validation pass. Both type checkers accept positive generic fixtures and reject
all seven deliberately invalid uses. Workflows pass actionlint 1.7.12; remote
GitHub Actions remains unexecuted.

MLP, TransformerOne and both pooling modes of ordinary/relation entity models
match original-source fixed CPU fixtures exactly under PyTorch 2.14.1: state
keys/tensors, evaluation/training predictions and all gradients. Existing full
model pickles and legacy/ZIP weight formats load. Actual read-only Chipiron
Morpion converters produce 93 × 25 tokens and 280 × 3 relations; both model
forwards/backwards match the original source exactly. This establishes tested
source-migration parity, without assuming bit-identical floating-point results
across different PyTorch versions or devices.

The prioritized follow-up work belongs partly to Chipiron training:

1. Measure mmap/meta/assign loading on realistic large artifacts, and reduce
   synchronization in Chipiron's per-batch timing/metric collection. The small
   loading probe improves median 10.54 to 3.69 ms with exact predictions; optimizer
   creation must follow assignment. Preserve old-format loading and recovery.
2. Evaluate supported GPU AMP, pinned transfers and optimizer implementation
   options in Chipiron, with numerical, validation-quality and resume checks.
   Zero-grad already uses modern set-to-none defaults; CUDA may already choose
   foreach. CPU foreach was slower in the isolated probe despite exact results.
3. Profile existing attention first. Entity MHA already requests no attention
   weights and can use SDPA. The additive-mask bypass remains necessary: a raw
   fused-encoder probe under 2.14.1 yielded nonfinite outputs. A dropout-zero
   legacy Head SDPA probe improved about 5%, with small forward/gradient
   differences; dropout/RNG and scientific validation remain separate work.
4. Keep compilation optional and experimental. The small MLP took 10.847 s to
   compile and then ran slower (16.61 versus 13.05 microseconds); entity fullgraph
   compilation hit data-dependent validation. No global backend was enabled.

The API constraints follow the official
[loading recipe](https://docs.pytorch.org/tutorials/recipes/recipes/module_load_state_dict_tips.html),
[attention documentation](https://docs.pytorch.org/docs/2.14/generated/torch.nn.MultiheadAttention.html),
[AMP documentation](https://docs.pytorch.org/docs/2.14/amp.html) and
[AdamW documentation](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html).
Chipiron production was inspected read-only, not migrated. Committed Morpion
training uses Adam; existing untracked work adds AdamW and schedules. Dataset
ordering, training recovery, optimizer/scheduler/RNG state and Python 3.14 worker
startup require dedicated subsequent migration tests.

Existing edits are preserved unchanged in `preserve/pre-python314-20261027`,
commit `30e487294066788dd29a51275b3fc93dc4feef74`, outside migration ancestry.
Additive-bias handling and persisted relation-scale configuration are already
incorporated upstream. Saved runtime overrides/diagnostics remain potentially
useful research features; current dirty Chipiron research code references those
absent APIs. Duplicate encoder implementations are obsolete; extra scale-boundary
tests remain useful candidates. No saved patch was automatically applied.

The tested preview is `algorhino-coral==0.1.16.dev0`, wheel SHA-256
`dce8b5d6cb8d0ae183bc0442cc087122c8168672d895cdfaee346ec3fd454876`.
All 24 packaged source/typing files match Git and the installed test environment.
Recommend final distribution **0.2.0** for Python 3.14-only support. Before
publishing, align Valanga metadata to published `0.3.0`, retain PyTorch >=2.14.1,
rebuild final-version wheels, rerun resolver/full release tests and obtain remote
CI/review. Exact requested foundation-commit wheels retain old version labels
and remain explicitly selected for migration validation. Coral does not bundle
Anemone; integration with its full validated Morpion implementation passed.

Full local evidence, immutable wheels, numerical fixtures, performance ranges,
preservation comparison and Chipiron recommendations are in
`/home/pompote/oldata/victor/python314-anemone-coral-0shyf5le/REPORT.md`.
