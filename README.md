# Breast cancer classification with PyTorch

This teaching project demonstrates a small end-to-end PyTorch prototype using
the Wisconsin Diagnostic Breast Cancer dataset. It covers data preparation,
custom `Dataset` and `DataLoader` classes, model construction, parameter
initialization, training, evaluation, and saving a model checkpoint.


## Project structure

```text
psi_practical_recap/
├── data/
│   ├── raw/                  # Original dataset
│   └── processed/            # Train, validation, and test CSV files
├── notebooks/
│   ├── data_exploration.ipynb
│   └── pytorch_modeling.ipynb
├── src/
│   ├── dataset.py            # Dataset and DataLoader definitions
│   ├── engine.py             # Training and evaluation loops
│   ├── model.py              # Network and Xavier initialization
│   ├── prepare_data.py       # Data preparation entry point
│   └── train.py              # Training entry point
├── outputs/                  # Generated model checkpoints
├── environment.yml
└── README.md
```

## Prerequisites

Install [Git](https://git-scm.com/) and
[Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) (or another
Conda-compatible distribution). A GPU is not required.

## 1. Copy the repository

Clone the repository and enter its directory:

```bash
git clone https://github.com/olimpijada72/psi_practical_recap.git
cd psi_practical_recap
```

Alternatively, download the repository as a ZIP file from GitHub, extract it,
and open a terminal in the extracted `psi_practical_recap` directory.

## 2. Set up the environment

Create the `psi_recap` environment from `environment.yml`:

```bash
conda env create --file environment.yml
```

Activate it:

```bash
conda activate psi_recap
```

You only need to create the environment once. Activate it again whenever you
open a new terminal and want to work on the project.

Verify that Python and PyTorch are available:

```bash
python --version
python -c "import torch; print(torch.__version__)"
```

If the environment already exists and `environment.yml` changes, update it with:

```bash
conda env update --name psi_recap --file environment.yml --prune
```

## 3. Prepare the data

The preparation script loads the dataset, creates stratified train, validation,
and test splits, fits a `RobustScaler` on the training split, and saves both the
raw and processed CSV files:

```bash
python src/prepare_data.py
```

The generated files are written to `data/raw/` and `data/processed/`.

## 4. Train and evaluate the model

Run the complete training workflow:

```bash
python src/train.py
```

The script loads the processed CSV files, trains the network, retains the epoch
with the best validation F1, evaluates that model on the test split, and writes
the checkpoint to `outputs/pytorch_classifier.pth`.

You can override common training settings:

```bash
python src/train.py --epochs 50 --batch-size 64 --learning-rate 0.0005
```

List every command-line option:

```bash
python src/train.py --help
```

## Complete command-line workflow

After cloning the repository, the complete workflow is:

```bash
conda env create --file environment.yml
conda activate psi_recap
python src/prepare_data.py
python src/train.py
```

Skip the first command if the environment has already been created.

## Run the notebooks

The notebooks explain the same workflow interactively. First register the Conda
environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name psi_recap --display-name "Python (psi_recap)"
```

Start JupyterLab from the repository root:

```bash
jupyter lab
```

Select **Python (psi_recap)** as the kernel and run the notebooks in order:

1. `notebooks/data_exploration.ipynb`
2. `notebooks/pytorch_modeling.ipynb`

The first notebook explores and prepares the data. The second walks through the
PyTorch prototype. Because the processed CSV files are included, the modeling
notebook can also be run independently.
