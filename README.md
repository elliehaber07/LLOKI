# Overview of LLOKI

![LLOKI recovered-5](https://github.com/user-attachments/assets/579abb41-2d58-49ca-928c-487f8deea9cc)

LLOKI is a novel framework designed for scalable spatial transcriptomics (ST) integration across diverse technologies without requiring shared gene panels. The framework consists of two key components:

1. **LLOKI-FP**: Utilizes optimal transport and feature propagation to perform a spatially informed transformation of ST gene expression profiles, aligning their sparsity with that of scRNA-seq. This optimizes the utility of **scGPT** embeddings.

2. **LLOKI-CAE**: A conditional autoencoder that integrates embeddings across ST technologies using a novel loss function. The loss function balances batch integration while preserving robust biological information from the LLOKI-FP embeddings.

This unique combination ensures the alignment of both features and batches, enabling robust ST data integration while preserving biological specificity and local spatial interactions.

---

## Running LLOKI

### Input Data Format

LLOKI requires spatial transcriptomics data in **AnnData** format. The data should be structured as follows:

#### 1. Spatial Coordinates
- The spatial coordinates of each cell should be included in the `.obsm` attribute of the AnnData object.
- Coordinates must be stored in `.obsm['spatial']` and formatted as an array with dimensions `[number of cells, 2]`, where each row represents the x and y coordinates of a cell.

#### 2. Gene Expression Data
- Gene expression data should be stored in `.X` as either a sparse or dense matrix with dimensions `[number of cells, number of genes]`.
  - Sparse matrices are recommended for large datasets.

#### 3. Additional Metadata (Optional)
- Any additional metadata (e.g., cell types, batch labels) can be stored in `.obs`.

---

## Installation

### Step 1: Create a Conda Environment

We recommend using **Anaconda** to manage your environment. If you haven't already, refer to the [Anaconda webpage](https://www.anaconda.com/) for installation instructions.

Create a Python 3.8 environment using the following command:

```bash
conda create --name lloki python=3.8
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

LLOKI consists of two key components: **LLOKI-FP** (Feature Propagation) and **LLOKI-CAE** (Conditional Autoencoder). The entire workflow can be executed using:

```bash
python run_lloki.py
```

This will:
1. Download the necessary data and model.
2. Run both parts of the pipeline.

### Components Breakdown

#### **LLOKI-FP (Feature Propagation)**
LLOKI-FP is responsible for transforming spatial transcriptomics (ST) data to align its sparsity with single-cell RNA sequencing (scRNA-seq) data. It does so using optimal transport and feature propagation methods.

Key functionalities:
- Reads spatial transcriptomics `.h5ad` files.
- Applies **spatial-aware batching** for large datasets.
- Propagates gene expression features using a reference scRNA-seq dataset.
- Embeds the processed data using **scGPT**.
- Saves the transformed data, which is then used by LLOKI-CAE.

To run LLOKI-FP separately, use:

```bash
python -c "from lloki.fp.run_lloki_fp import run_lloki_fp; run_lloki_fp(args)"
```

#### **LLOKI-CAE (Conditional Autoencoder)**
LLOKI-CAE performs integration of the transformed ST embeddings while preserving biological variation across different technologies. It employs a **conditional autoencoder** with:
- **Triplet loss** for feature alignment.
- **Neighborhood preservation loss** for local spatial consistency.
- **Batch correction** to reduce dataset-specific biases.

Key functionalities:
- Reads processed `.h5ad` files from LLOKI-FP.
- Merges datasets while preserving local spatial structure.
- Learns a **latent representation** using a trained conditional autoencoder.
- Outputs **integrated embeddings** that can be used for downstream analysis.

To run LLOKI-CAE separately, use:

```bash
python -c "from lloki.cae.run_lloki_cae import run_lloki_cae; run_lloki_cae(args)"
```

---

## Parameters and Configuration

LLOKI provides various tunable parameters to adjust the integration process. These can be modified when running `run_lloki.py` via command-line arguments.

| Parameter | Description | Default |
|-----------|------------|---------|
| `--data_dir` | Directory for input spatial transcriptomics data | `data/input_slices` |
| `--output_dir` | Directory for saving outputs | `output` |
| `--model_dir` | Path to scGPT model | `external/scgpt` |
| `--reference_data_path` | Path to scRNA-seq reference data | `data/reference_data/scref_full.h5ad` |
| `--checkpoint_dir` | Directory for model checkpoints | `checkpoints` |
| `--k` | Number of neighbors for KNN in LLOKI-FP | `40` |
| `--iter` | Number of iterations for feature propagation | `40` |
| `--alpha` | Weighting parameter for propagation | `0.05` |
| `--seed` | Random seed for reproducibility | `0` |
| `--device` | Device to run computations (`cuda` or `cpu`) | `cuda` |
| `--npl_num_neighbors` | Number of neighbors for neighborhood preservation loss in LLOKI-CAE | `30` |
| `--lambda_neighborhood` | Weighting factor for neighborhood preservation loss | `500` |
| `--lambda_triplet` | Weighting factor for triplet loss | `2` |
| `--lr` | Learning rate for training LLOKI-CAE | `0.005` |
| `--epochs` | Number of epochs for training LLOKI-CAE | `200` |
| `--batch_size` | Batch size for training LLOKI-CAE | `16000` |
| `--batch_dim` | Dimensionality of batch embeddings | `10` |
| `--num_batches` | Number of spatial technologies integrated | `5` |

---

## Output Files

After running LLOKI, the output directory will contain:

- **Processed ST Data (`*_processed.h5ad`)**: Transformed ST datasets with embedded representations.
- **Trained LLOKI-CAE Model (`checkpoints/`)**: Model weights and logs for integration.
- **Final Integrated Data (`cae_umap.png`)**: UMAP visualization of the integrated spatial transcriptomics data.

---

## Citation

If you use LLOKI in your research, please cite:

> **Your Paper or Preprint (if available)**  
> Authors, *Title*, Year.

---

## Contact

For issues, questions, or contributions, please reach out or submit an issue on [GitHub](https://github.com/your-repo).

---
