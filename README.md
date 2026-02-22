# deep-jsense
Code for Deep J-Sense: Accelerated MRI Reconstruction via Unrolled Alternating Optimization

# Pre-requisites
Install [`uv`](https://docs.astral.sh/uv/getting-started/installation/).

# Training on FastMRI knee scans
Run `uv run train.py --data_dir /path/to/fastMRI`.

The passed directory must contain the `multicoil_train` and `multicoil_val` sub-directories.

# Inference on FastMRI knee scans
Run `uv run inference.py --data_dir /path/to/fastMRI`.

The passed directory must contan the `multicoil_val` sub-directory.