# Breast cancer classification with PyTorch

This is a small, teaching-oriented machine-learning project built around the
Wisconsin Diagnostic Breast Cancer dataset. It uses numerical measurements of
cell nuclei to train a neural network that classifies samples as benign or
malignant.

The notebooks introduce the complete workflow in two stages:

1. `data_exploration.ipynb` explores and prepares the data.
2. `pytorch_modeling.ipynb` teaches the core PyTorch workflow, with an emphasis
   on custom `Dataset` classes, `DataLoader`s, neural networks, training loops,
   evaluation, and saving model parameters.



## Repository structure

```text
psi_practical_recap/
├── data/
│   ├── raw/
│   │   └── breast_cancer_data.csv
│   └── processed/
│       ├── train.csv
│       ├── validation.csv
│       └── test.csv
├── notebooks/
│   ├── data_exploration.ipynb
│   └── pytorch_modeling.ipynb
├── src/
│   ├── dataset.py            # Dataset and DataLoader definitions
│   ├── engine.py             # Training and evaluation loops
│   ├── model.py              # Network and parameter initialization
│   ├── prepare_data.py       # Rebuilds the raw and processed CSV files
│   └── train.py              # Runs the complete training workflow
├── outputs/                  # Generated model files and predictions
├── environment.yml
└── README.md
```

## Prerequisites

Install the following before starting:

- [Git](https://git-scm.com/)
- [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) or another
  Conda-compatible distribution

The supplied environment uses Python 3.11 and installs CPU PyTorch, pandas,
NumPy, scikit-learn, Matplotlib, JupyterLab, and ipykernel. A GPU is not required.

## 1. Clone the repository

```bash
git clone https://github.com/olimpijada72/psi_practical_recap.git
cd psi_practical_recap
```

## 2. Create the Conda environment

Create the environment from the checked-in specification:

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate psi_recap
```


## 3. Register the Jupyter kernel

With `psi_recap` activated, register it as a notebook kernel:

```bash
python -m ipykernel install --user --name psi_recap --display-name "Python (psi_recap)"
```

This command normally needs to be run only once. In JupyterLab or VS Code,
select **Python (psi_recap)** as the kernel for each notebook.


## 4. Run the notebooks interactively

Start JupyterLab from the repository root:

```bash
jupyter lab
```

Open and run the notebooks in this order:

1. `notebooks/data_exploration.ipynb`
2. `notebooks/pytorch_modeling.ipynb`

In each notebook, confirm that **Python (psi_recap)** is selected, then use
**Run > Run All Cells**. The processed CSV files are already included, so the
modeling notebook can also be run by itself.

The modeling notebook writes generated artifacts to `outputs/`, including the
trained model parameters. Generated output files are ignored by Git.

## Prepare the data without Jupyter

The essential preparation workflow is also available as a standalone script.
It loads the scikit-learn dataset, creates stratified 60/20/20 splits, fits a
`RobustScaler` only on the training data, and saves both the raw and processed
CSV files:

```bash
python src/prepare_data.py
```

Run this command with `psi_recap` activated. The script resolves paths from its
own location, so it can be invoked from any working directory.

## Train without Jupyter

The notebook prototype is divided into small modules under `src/`. Run the
complete training and evaluation workflow from the repository root with:

```bash
python src/train.py
```

The script trains for 30 epochs by default, retains the epoch with the highest
validation F1, evaluates it on the test split, and saves the checkpoint to
`outputs/pytorch_classifier.pth`.

Common settings can be changed from the command line:

```bash
python src/train.py --epochs 50 --batch-size 64 --learning-rate 0.0005
```

Display every available option with:

```bash
python src/train.py --help
```
