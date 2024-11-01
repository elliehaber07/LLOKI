# Overview of LLOKI

<img width="649" alt="Screenshot 2024-11-01 at 12 34 29 AM" src="https://github.com/user-attachments/assets/5ab3fa0f-91c7-428e-9085-25df9e1cf321">

LLOKI is a novel framework for scalable spatial transcriptomics (ST) integration across diverse technologies without requiring shared gene panels. The framework comprises two key components: 
LLOKI-FP, which leverages optimal transport and feature propagation to perform a spatially informed transformation of ST gene expression profiles, aligning their sparsity with that of scRNA-seq to optimize the utility of scGPT embeddings; and LLOKI-CAE, a conditional autoencoder that integrates embeddings across ST technologies using a novel loss function that balances batch integration with the preservation of robust biological information from the LLOKI-FP embeddings. This unique combination ensures alignment of both features and batches, enabling robust ST data integration while preserving biological specificity and local spatial interactions.

# Running LLOKI
# Input Data Format

LLOKI requires spatial transcriptomics data to be provided as AnnData objects, structured as follows:

Spatial Coordinates:
The spatial coordinates of each cell should be included in the .obsm attribute of the AnnData object.
The coordinates must be stored under .obsm['spatial'] and formatted as an array with dimensions [number of cells, 2] (representing x and y coordinates for each cell)
Gene Expression Data:
Gene expression data should be stored in .X as a sparse matrix (recommended for large datasets) or a dense matrix, with dimensions [number of cells, number of genes]

While not required, any additional metadata (e.g., cell types, batch labels) can be stored in .obs
