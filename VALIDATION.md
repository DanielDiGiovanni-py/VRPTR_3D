# Release checks

Checked on 30 September 2026.

- All 15 existing CPU regression tests pass. They cover the fixed training template, provenance checks, residual matching, missing support, ties, selective prediction, component-swap metrics, and figure geometry.
- All Python files compile and all shell scripts pass syntax checks. Retained internal imports resolve.
- Eight shell workflow scenarios were exercised with a stub interpreter, including paths containing spaces. Their generated options match the Python command-line parsers. These checks verify command construction, not model execution.
- The model syntax tree is unchanged from the September source snapshot. Retained loss, metric, and shared statistical function bodies are unchanged. Data-root overrides operate on local configuration copies and do not rewrite checkpoints.
- Preprocessing command help, import safety, and dry-run behavior were checked. The original Nipype graph settings and connections were compared, and numerical helper behavior was compared on synthetic inputs with dependency stand-ins.

Run the included regression tests from the project root:

```bash
python -m unittest discover -s tests -v
```

No full training, checkpoint-based inference, or study-image preprocessing was rerun for this release. PyTorch, NiBabel, Nilearn, the original images, and trained checkpoints were not available in the validation environment. These checks support a conservative source cleanup; they do not independently reproduce the paper's results.

The exact historical comparator fits, cohort-selection decisions, and manual quality-control records still require the authors' original run records. Available evidence and the archived training recipe are described in [docs/reproducibility.md](docs/reproducibility.md).
