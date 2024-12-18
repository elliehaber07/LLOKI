import torch
import torch.nn.functional as F
import numpy as np


def top_k(raw_graph, K):
    """Select top K neighbors for each node and mask others."""
    values, indices = raw_graph.topk(k=K, dim=-1)
    mask = torch.zeros_like(raw_graph, device=raw_graph.device)
    mask.scatter_(1, indices, 1.0)  # Mask non-top-K values
    return raw_graph * mask


def post_processing(cur_raw_adj, add_self_loop=True, sym=True, gcn_norm=False):
    """Post-process adjacency matrix with optional self-loops, symmetrization, and GCN normalization."""
    if add_self_loop:
        cur_raw_adj += torch.eye(cur_raw_adj.size(0), device=cur_raw_adj.device)
    if sym:
        cur_raw_adj = (cur_raw_adj + cur_raw_adj.t()) / 2
    deg = cur_raw_adj.sum(1)
    deg_inv_sqrt = deg.pow(-0.5 if gcn_norm else -1).masked_fill(deg == 0, 0.0)
    return deg_inv_sqrt.diag() @ cur_raw_adj @ deg_inv_sqrt.diag()


def knn_graph(embeddings, k, gcn_norm=False, sym=True):
    """Generate KNN graph from embeddings with optional GCN normalization and symmetrization."""
    embeddings = F.normalize(embeddings, dim=1)
    similarity_graph = top_k(torch.mm(embeddings, embeddings.t()), k + 1).relu()
    sparse_adj = post_processing(
        similarity_graph, gcn_norm=gcn_norm, sym=sym
    ).to_sparse()
    return sparse_adj.indices(), sparse_adj.values()


def apply_gcn_normalization(adj_matrix):
    """Normalize adjacency matrix with GCN normalization."""
    deg_inv_sqrt = adj_matrix.sum(1).pow(-0.5).masked_fill(adj_matrix.sum(1) == 0, 0.0)
    return deg_inv_sqrt.diag() @ adj_matrix @ deg_inv_sqrt.diag()


def create_spatially_weighted_knn_graph(
    embeddings, adata, k, device, gcn_norm=False, sym=True
):
    """Refine KNN graph by spatial distances, apply optional symmetrization and GCN normalization."""
    knn_index, knn_weight = knn_graph(embeddings, k, gcn_norm=gcn_norm, sym=sym)
    knn_adj = torch.zeros((embeddings.size(0),) * 2, device=device)
    knn_adj[knn_index[0], knn_index[1]] = knn_weight

    refined_adj = torch.zeros_like(knn_adj)
    spatial_coords = torch.tensor(adata.obsm["spatial"], device=device)
    avg_dist = torch.cdist(
        spatial_coords[np.random.choice(spatial_coords.size(0), 100)],
        spatial_coords[np.random.choice(spatial_coords.size(0), 100)],
    ).mean()

    # Refine neighbors by spatial distance
    for i in range(embeddings.size(0)):
        neighbors = knn_index[1][knn_index[0] == i]
        if neighbors.numel():
            distances = (spatial_coords[neighbors] - spatial_coords[i]).norm(dim=1)
            refined_adj[i, neighbors] = torch.exp(
                -distances.square() / (2 * avg_dist**2)
            ).float()
            refined_adj[i, i] = 1.0

    # Symmetrize and normalize
    combined_adj = (refined_adj + refined_adj.t()) / 2 if sym else refined_adj
    if gcn_norm:
        combined_adj = apply_gcn_normalization(combined_adj)

    # Convert to sparse and extract indices and weights
    combined_sparse = combined_adj.to_sparse().coalesce()
    return combined_sparse.indices(), combined_sparse.values()
