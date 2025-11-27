# NN - Neural Networks (Course scaffold)

Neural Networks: Theory and Applications (Prof. Dr. Stefanie Vogl)

This repository contains course materials for the "Neural Networks: Theory and Applications" from OTH Regensburg.

## Structure

- `lectures/` — lecture slides and notes
- `exercises/` — practical exercises and assignments
- `projects/` — student projects and reports
- `notebooks/` — Jupyter notebooks and guided tutorials
- `code/` — runnable example scripts (Keras / TensorFlow / PyTorch)
- `data/` — datasets used in exercises (gitignored by default)
- `references/` — papers and reading list
- `docs/` — additional documentation

## NN - Neural Networks (Course scaffold)

Neural Networks: Theory and Applications (Prof. Dr. Stefanie Vogl)

This repository contains lecture material, exercises, notebooks and example code for an introductory neural networks course.

### Quick start (minimal)

1. Create and activate a virtual environment (from repository root):

```bash
# from repository root (macOS / zsh)
python3 -m venv .venv
source .venv/bin/activate
```

1. Install dependencies (inside the virtual environment):

```bash
python -m pip install --upgrade pip
pip install -r requirements-macos.txt
```

3. Install PyTorch with Metal (if on Apple Silicon):

```bash
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

4. Verify PyTorch + TensorFlow (Metal) availability:

Run the provided test script for PyTorch:

```bash
python code/test_pytorch.py
```

And for TensorFlow:

```bash
python code/test_tensorflow.py
```

1. Open the notebook server and run the example:

```bash
jupyter lab
```

Open `notebooks/simple_mlp.ipynb` and run the cells.

TensorFlow (optional)

If you want TensorFlow on macOS, follow the official guide:
https://www.tensorflow.org/install/pip#macos

Files in this repo

- `notebooks/` — Jupyter notebooks (start with `simple_mlp.ipynb`).
- `code/` — runnable examples (currently `code/test_pytorch.py`).
- `data/` — example datasets (CSV files).
- `requirements-macos.txt` — pip-style list for macOS (mentions PyTorch entries).
- `requirements-pytorch.txt` — minimal PyTorch package list.
