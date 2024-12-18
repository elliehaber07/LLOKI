import torch
from lloki.fp.feature_propagation import FeaturePropagation
from lloki.fp.graph_construction import create_spatially_weighted_knn_graph
import numpy as np
from scipy.sparse import issparse
from scipy.interpolate import interp1d


def calculate_sparsity(adata):
    """Calculate sparsity per cell for an AnnData object."""
    return (
        1 - (adata.X.getnnz(axis=1) / adata.X.shape[1])
        if issparse(adata.X)
        else np.mean(adata.X == 0, axis=1)
    ).flatten()


def compute_empirical_cdf(sparsity_values):
    """Compute the empirical CDF for sorted sparsity values."""
    sorted_vals = np.sort(sparsity_values)
    return sorted_vals, np.linspace(0, 1, len(sorted_vals))


def map_sparsity_values(cdf_src, sorted_src, cdf_tgt, sorted_tgt):
    """Map sparsity values from source to target distribution using interpolation."""
    return interp1d(
        cdf_tgt,
        sorted_tgt,
        bounds_error=False,
        fill_value=(sorted_tgt[0], sorted_tgt[-1]),
    )(cdf_src)


def propagate(adata, adata_scrna, args):
    device = args.device
    adata = adata[:, adata.var_names.isin(adata_scrna.var_names)]
    cell_data = adata.X.toarray() if issparse(adata.X) else adata.X.copy()
    sparsity_scrna = calculate_sparsity(adata_scrna)
    sparsity_merfish = calculate_sparsity(adata)

    # OT mapping
    sorted_sparsity_scrna, cdf_scrna = compute_empirical_cdf(sparsity_scrna)
    sorted_sparsity_merfish, cdf_merfish = compute_empirical_cdf(sparsity_merfish)
    mapped_sparsity = map_sparsity_values(
        cdf_merfish, sorted_sparsity_merfish, cdf_scrna, sorted_sparsity_scrna
    )
    target_sparsity = mapped_sparsity[np.argsort(sparsity_merfish)]

    # Non-zero entries adjustment
    num_genes = adata.X.shape[1]
    existing_nonzero = (
        adata.X.getnnz(axis=1)
        if issparse(adata.X)
        else np.count_nonzero(adata.X, axis=1)
    )
    max_impute_per_cell = torch.tensor(
        np.clip(np.ceil((1 - target_sparsity) * num_genes) - existing_nonzero, 0, None),
        device=device,
        dtype=torch.int32,
    )

    # Normalize data and create KNN graph for hard propagation
    cell_data = torch.Tensor(cell_data).to(device)
    cell_data = cell_data / (torch.max(cell_data) + 1e-8)
    edge_index, edge_weight = create_spatially_weighted_knn_graph(
        cell_data, adata, args.k, device, True, True
    )

    model = FeaturePropagation(
        num_iterations=args.iter,
        adata=adata,
        mask=True,
        alpha=0.0,
        max_imputation_per_cell=max_impute_per_cell,
        device=device,
    ).to(device)
    denoised_matrix = model(cell_data, edge_index.to(device), edge_weight.to(device))
    adata.obsm["denoised"] = denoised_matrix.cpu().numpy()

    # Soft propagation with new graph
    print("Starting Soft Feature Propagation...")
    edge_index, edge_weight = create_spatially_weighted_knn_graph(
        denoised_matrix, adata, args.k, device, True, True
    )
    model = FeaturePropagation(
        num_iterations=1, adata=adata, mask=False, alpha=1, device=device
    ).to(device)

    denoised_matrix=model(cell_data, edge_index.to(device), edge_weight.to(device))
    adata.obsm['denoised'] = denoised_matrix.detach().cpu().numpy()

    return adata
    
