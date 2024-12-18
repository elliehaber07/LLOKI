# Overview of LLOKI

![LLOKI Framework](https://github.com/user-attachments/assets/5ab3fa0f-91c7-428e-9085-25df9e1cf321)

LLOKI is a novel framework designed for scalable spatial transcriptomics (ST) integration across diverse technologies without requiring shared gene panels. The framework consists of two key components:

1. **LLOKI-FP**: Utilizes optimal transport and feature propagation to perform a spatially informed transformation of ST gene expression profiles, aligning their sparsity with that of scRNA-seq. This optimizes the utility of **scGPT** embeddings.

2. **LLOKI-CAE**: A conditional autoencoder that integrates embeddings across ST technologies using a novel loss function. The loss function balances batch integration while preserving robust biological information from the LLOKI-FP embeddings.

This unique combination ensures the alignment of both features and batches, enabling robust ST data integration while preserving biological specificity and local spatial interactions.

---

# Running LLOKI

## Input Data Format

LLOKI requires spatial transcriptomics data in **AnnData** format. The data should be structured as follows:

### 1. Spatial Coordinates
- The spatial coordinates of each cell should be included in the `.obsm` attribute of the AnnData object.
- Coordinates must be stored in `.obsm['spatial']` and formatted as an array with dimensions `[number of cells, 2]`, where each row represents the x and y coordinates of a cell.

### 2. Gene Expression Data
- Gene expression data should be stored in `.X` as either a sparse or dense matrix with dimensions `[number of cells, number of genes]`.
  - Sparse matrices are recommended for large datasets.

### 3. Additional Metadata (Optional)
- Any additional metadata (e.g., cell types, batch labels) can be stored in `.obs`.

---

## Installation

### Step 1: Create a Conda Environment

We recommend using **Anaconda** to manage your environment. If you haven't already, refer to the [Anaconda webpage](https://www.anaconda.com/) for installation instructions.

Create a Python 3.8 environment using the following command:

```bash
conda install --name lloki python=3.8
```

Activate the environment:

```bash
conda activate lloki
```

### Step 2: Install Dependencies

#### Install PyTorch with CUDA (Optional)
If you have an NVIDIA GPU and want to use CUDA for acceleration, install PyTorch with the desired CUDA version. For example, to install PyTorch 2.1.0 with CUDA 11.8:

```bash
conda install pytorch==2.1.0 cudatoolkit=11.8 -c pytorch
```

For a CPU-only installation, simply omit the `cudatoolkit` argument.

#### Install Remaining Dependencies
Install the remaining required packages using the provided `requirements.txt` file:

```bash
pip install -r requirements.txt
```

---

## Running the Code

The method consists of two parts: **LLOKI-FP** and **LLOKI-CAE**. To run both parts, use the following command:

```bash
python run_lloki.py
```

This will:
1. Download the necessary data and model.
2. Run both parts of the pipeline.

---
