import torch


class FeaturePropagation(torch.nn.Module):
    def __init__(
        self,
        num_iterations,
        adata,
        mask,
        alpha=0.0,
        max_imputation_per_cell=None,
        early_stopping=True,
        patience=15,
        device="cuda",
    ):
        super(FeaturePropagation, self).__init__()
        self.num_iterations = num_iterations
        self.mask = mask
        self.alpha = alpha
        self.early_stopping = early_stopping
        self.patience = patience
        self.adata = adata
        self.max_imputation_per_cell = max_imputation_per_cell
        self.device = device

    def forward(self, x, edge_index, edge_weight):
        device = self.device
        x = x.to(device)
        edge_index = edge_index.to(device)
        edge_weight = edge_weight.to(device)

        original_x = x.clone()
        n_nodes, n_genes = x.shape

        # Residual term for the propagation formula
        res = (1 - self.alpha) * x
        adj = (
            torch.sparse.FloatTensor(edge_index, edge_weight, size=(n_nodes, n_nodes))
            .to(device)
            .float()
        )

        # Differentiate processing for max_imputation_per_cell
        if self.max_imputation_per_cell is not None:
            existing_nonzero_mask = x != 0
            imputation_mask = torch.zeros_like(x, dtype=torch.bool, device=device)
            max_imputation_per_cell = self.max_imputation_per_cell.to(device)

        for i in range(self.num_iterations):
            previous_x = x.clone()
            x = torch.sparse.mm(adj, x)  # Feature propagation step

            if self.mask:
                if self.max_imputation_per_cell is not None:
                    x[existing_nonzero_mask] = original_x[existing_nonzero_mask]
                    newly_imputed = (x != 0) & ~existing_nonzero_mask & ~imputation_mask
                    imputation_mask |= newly_imputed
                    total_imputed_per_cell = imputation_mask.sum(dim=1)
                    over_imputed_cells = (
                        total_imputed_per_cell > max_imputation_per_cell
                    )

                    if over_imputed_cells.any():
                        for cell_idx in over_imputed_cells.nonzero(as_tuple=True)[0]:
                            excess = (
                                total_imputed_per_cell[cell_idx]
                                - max_imputation_per_cell[cell_idx]
                            ).item()
                            imputed_gene_indices = imputation_mask[cell_idx].nonzero(
                                as_tuple=True
                            )[0]
                            genes_to_reset = imputed_gene_indices[
                                torch.randperm(len(imputed_gene_indices))[:excess]
                            ]
                            x[cell_idx, genes_to_reset] = 0
                            imputation_mask[cell_idx, genes_to_reset] = False
                else:
                    nonzero_idx = torch.nonzero(original_x)
                    nonzero_i, nonzero_j = nonzero_idx.t()
                    x[nonzero_i, nonzero_j] = original_x[nonzero_i, nonzero_j]
            else:
                x.mul_(self.alpha).add_(res)

            # Check for early stopping, if enabled
            if self.early_stopping and torch.norm(x - previous_x) < self.patience:
                break

        return x
