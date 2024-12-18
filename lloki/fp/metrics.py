import scanpy as sc
import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
import matplotlib.pyplot as plt


def leiden_cluster(
    dataset, use_rep, target_key="label", resolution=1.0, tolerance=0.01
):
    """
    Perform Leiden clustering, adjusting resolution to match target clusters.
    """
    sc.pp.neighbors(dataset, use_rep=use_rep)
    sc.tl.leiden(dataset, resolution=resolution)
    target_clusters = dataset.obs[target_key].nunique()
    lower, upper = 0.1 * resolution, 10 * resolution

    while target_clusters and np.abs(lower - upper) > tolerance:
        resolution = (lower * upper) ** 0.5
        sc.tl.leiden(dataset, resolution=resolution)
        num_clusters = dataset.obs["leiden"].nunique()
        lower, upper = (
            (resolution, upper)
            if num_clusters < target_clusters
            else (lower, resolution)
        )
        if num_clusters == target_clusters:
            break

    print(f"Clusters: {num_clusters}, Resolution: {resolution:.3f}")


def get_clustering_scores(
    imputed, embedding_key, keep_obs="label", ct_threshold=0.01, resolution=1.0, filename=None
):
    """
    Compute clustering metrics and plot UMAP for imputed data.
    """
    sc.pp.neighbors(imputed, use_rep=embedding_key)
    sc.tl.umap(imputed)

    # Filter subclasses with sufficient representation
    min_count = int(len(imputed) * ct_threshold)
    subclasses = imputed.obs[keep_obs].value_counts()
    imputed_sub = imputed[
        imputed.obs[keep_obs].isin(subclasses[subclasses > min_count].index)
    ].copy()

    leiden_cluster(imputed_sub, use_rep=embedding_key, resolution=resolution)

    # Plot UMAP and save
    sc.pl.umap(
        imputed_sub,
        color=[keep_obs, "leiden"],
        title=[f"{keep_obs}", "Leiden Clusters"],
    )
    plt.savefig(f"output/imputed_umap_{filename}.png")

    # Calculate and print clustering scores
    ari = adjusted_rand_score(imputed_sub.obs[keep_obs], imputed_sub.obs["leiden"])
    nmi = normalized_mutual_info_score(
        imputed_sub.obs[keep_obs], imputed_sub.obs["leiden"]
    )
    asw = silhouette_score(imputed_sub.obsm[embedding_key], imputed_sub.obs[keep_obs])
    print(f"ARI: {ari:.3f}, NMI: {nmi:.3f}, ASW: {asw:.3f}")
