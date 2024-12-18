import glob
import os
import anndata as ad
import numpy as np
import scanpy as sc
import torch
from lloki.fp.lloki_fp import propagate
from lloki.fp.metrics import get_clustering_scores
from lloki.utils import set_seed
import scgpt as scg

def run_lloki_fp(args):
    """
    Run Lloki propagation on spatial transcriptomics data files, perform batching and
    embedding, and save processed data.
    """
    # Limit CPU threads for PyTorch
    torch.set_num_threads(3)

    # Load reference data for propagation
    adata_scrna = sc.read_h5ad(args.reference_data_path)  # Update path

    # Process each .h5ad file in the data directory
    for h5ad_file in glob.glob(os.path.join(args.data_dir, "*.h5ad")):
        print(f"Processing file: {h5ad_file}")
        adata = ad.read_h5ad(h5ad_file)
        set_seed(args.seed)  # Ensure reproducibility
        batches = [adata]  # Default to single batch

        # Apply spatially aware batching if large data
        if adata.shape[0] >= 30000:
            sorted_indices = np.argsort(adata.obsm["spatial"][:, 0])
            num_splits = 4 if "xenium" in h5ad_file else 2  # More splits for 'xenium' files
            batches = [adata[s] for s in np.array_split(sorted_indices, num_splits)]
        
        # Propagate each batch and combine results
        adata = ad.concat([propagate(batch, adata_scrna, args) for batch in batches], join="outer", index_unique=None)
        adata.obsm["spatial"] = np.vstack([b.obsm["spatial"] for b in batches])
        adata = adata[:, adata.var_names.isin(adata_scrna.var_names)]  # Filter genes present in reference data

        # Copy data for embedding and prepare necessary attributes
        adata_copy = adata.copy()
        adata_copy.X = adata.obsm["denoised"].toarray()
        adata_copy.var["gene_names"] = [v.upper() for v in adata_copy.var_names]

        # Perform embedding with scGPT
        processed_data = scg.tasks.embed_data(adata_copy, args.model_dir, batch_size=64, gene_col="gene_names")

        # Save processed AnnData with embedding to the output directory
        basename = os.path.splitext(os.path.basename(h5ad_file))[0]
        output_path = os.path.join(args.output_dir, f"{basename}_processed.h5ad")
        processed_data.write_h5ad(output_path)
        print(f"Saved: {output_path}\nScore: {get_clustering_scores(processed_data, 'X_scGPT', filename=basename)}")
