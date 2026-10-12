# Contributing to TFG-Net

Thanks for your interest in improving TFG-Net!

## Development setup

```bash
git clone https://github.com/gejingjun/Tri-Net-urban-target-detect.git
cd Tri-Net-urban-target-detect
conda env create -f environment.yml   # or: pip install -r requirements.txt
conda activate tfgnet
pip install -e .                      # editable install of src/tfgnet
```

## Running the tests

The whole test suite runs on CPU synthetic data in a few minutes:

```bash
pytest
```

For a quick smoke test only:

```bash
pytest -m "not slow"
```

## Code style

- 4-space indentation, PEP 8.
- Every module that implements an equation from the paper carries the
  equation number in its docstring (e.g. `Eq. 7`, `Eq. 19`) so reviewers can
  trace code ↔ paper.
- New experiment configs go into `experiments/configs/` and must be reachable
  from `scripts/run_all_experiments.py`.

## Workflow

1. Branch from `main`: `git checkout -b feat/your-idea`.
2. Keep changes focused; add or update tests under `tests/`.
3. Run `pytest` locally before opening a pull request.
4. Describe reproducible commands in the PR body if you add new scripts.

## Reporting issues

Please include: commit hash, config file, `python -c "import torch; print(torch.__version__)"`
output, and the exact command you ran.
