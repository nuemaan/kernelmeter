# Contributing

Setup:

```bash
git clone https://github.com/nuemaan/kernelmeter
cd kernelmeter
pip install -e ".[dev]"
pytest
```

The tests fake libcuda and libnvidia-ml, so everything runs without a GPU.

## The house rules

- **Numbers must be checkable.** Database entries store physical parameters
  (SMs, clocks, bus width, per-pin rate, vram), never headline numbers, and
  every entry needs a test asserting the derived bandwidth and fp32 tflops
  against the vendor spec sheet. If you can't cross-check a number, leave
  the card out (see the b200 issue for an example).
- **Copies are drift-tested.** The web demo embeds the gpu database and the
  quant table. After touching `gpus.py`, run
  `python scripts/update_webdb.py`; the test suite fails until you do.
- **The readme shows real output.** If a change alters what a command
  prints, update the sample in the readme with actual output, not an
  approximation.

## Easy first contributions

Adding a card to `src/kernelmeter/gpus.py` is one line plus one test line.
Grab the physical parameters from the vendor page or techpowerup, check
that the derived numbers match the published ones, and open a PR.

## Releases (maintainer)

Bump the version in `pyproject.toml` and `src/kernelmeter/__init__.py`,
add a changelog entry, push a `v*` tag. CI tests, builds and publishes to
PyPI through trusted publishing.
