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

The model is selected using validation F1. The modeling notebook also contains
an exercise that changes the selection objective to validation accuracy.

> This repository is an educational example, not a clinical diagnostic system.

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

If the environment already exists and `environment.yml` has changed, update it
instead:

```bash
conda env update --name psi_recap --file environment.yml --prune
```

## 3. Register the Jupyter kernel

With `psi_recap` activated, register it as a notebook kernel:

```bash
python -m ipykernel install --user --name psi_recap --display-name "Python (psi_recap)"
```

This command normally needs to be run only once. In JupyterLab or VS Code,
select **Python (psi_recap)** as the kernel for each notebook.

To verify that the environment is active and PyTorch is available:

```bash
python -c "import torch; print(torch.__version__)"
```

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

## Run everything from the command line

With the environment activated and from the repository root, execute both
notebooks in place:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/data_exploration.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/pytorch_modeling.ipynb
```

The second command should be run only after the first has completed. Executing
with `--inplace` saves cell outputs back into each notebook.

## Start again with a clean environment

If the environment becomes inconsistent, remove and recreate it:

```bash
conda deactivate
conda env remove --name psi_recap
conda env create -f environment.yml
conda activate psi_recap
```

The Jupyter kernel registration can remain in place because it points to the
environment name. If needed, register it again using the command in step 3.
