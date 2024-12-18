import os
import torch
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np


def compute_batch_nearest_neighbors(batchdata, embeddings, k_neighbors=30):
    """Compute nearest neighbors for each batch independently."""
    all_neighbor_indices, all_neighbor_distances = [], []
    for batch in torch.unique(batchdata):
        batch_indices = torch.where(batchdata == batch)[0].cpu().numpy()
        batch_embeddings = embeddings[batch_indices].cpu().numpy()
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(batch_embeddings)
        distances, indices = nbrs.kneighbors(batch_embeddings)
        all_neighbor_indices.append(batch_indices[indices[:, 1:]])
        all_neighbor_distances.append(distances[:, 1:])
    return np.concatenate(all_neighbor_indices), np.concatenate(all_neighbor_distances)


def create_chunks(device, batch_labels, num_chunks):
    """Split data into balanced chunks based on batch labels."""
    unique_batches = torch.unique(batch_labels).tolist()
    batch_indices = {b: torch.where(batch_labels == b)[0] for b in unique_batches}
    for b in unique_batches:
        batch_indices[b] = batch_indices[b][torch.randperm(batch_indices[b].size(0))]
    chunks = [
        torch.empty(0, dtype=torch.long, device=device) for _ in range(num_chunks)
    ]
    for b in unique_batches:
        num_samples_per_chunk = batch_indices[b].size(0) // num_chunks
        remainder = batch_indices[b].size(0) % num_chunks
        start_idx = 0
        for i in range(num_chunks):
            end_idx = start_idx + num_samples_per_chunk
            chunks[i] = torch.cat((chunks[i], batch_indices[b][start_idx:end_idx]))
            start_idx = end_idx
        if remainder:
            remainder_indices = batch_indices[b][start_idx:]
            random_chunk_indices = torch.randperm(remainder)
            for i, remainder_idx in enumerate(random_chunk_indices):
                chunks[i] = torch.cat(
                    (chunks[i], remainder_indices[remainder_idx : remainder_idx + 1])
                )
    return chunks


def find_mutual_nearest_neighbors(embeddings_list, n_neighbors=50):
    """Find mutual nearest neighbors across batches."""
    mutual_pairs = []
    for batch_i in range(len(embeddings_list)):
        for batch_j in range(batch_i + 1, len(embeddings_list)):
            knn_A_to_B = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine").fit(
                embeddings_list[batch_j]
            )
            distances_A_to_B, indices_A_to_B = knn_A_to_B.kneighbors(
                embeddings_list[batch_i]
            )
            knn_B_to_A = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine").fit(
                embeddings_list[batch_i]
            )
            distances_B_to_A, indices_B_to_A = knn_B_to_A.kneighbors(
                embeddings_list[batch_j]
            )
            for i, neighbors_in_B in enumerate(indices_A_to_B):
                for neighbor_in_B in neighbors_in_B:
                    if i in indices_B_to_A[neighbor_in_B]:
                        mutual_pairs.append(
                            {
                                "batch_A": batch_i,
                                "index_A": i,
                                "batch_B": batch_j,
                                "index_B": neighbor_in_B,
                            }
                        )
                        break
    print(f"Mutual pairs found: {len(mutual_pairs)}")
    return mutual_pairs


def train_autoencoder_mnn_triplet_prechunk(
    model,
    args,
    data,
    lr,
    epochs,
    evaluate_interval=1,
    lamb=0.5,
    lamb_neighborhood=15.0,
    pretrain_epochs=50,
    margin=1.0,
    knn=100,
    update_interval=15,
    chunk_size=512,
    verbose=False,
    checkpoint_interval=25,
    checkpoint_start=0,
    ramp_up_epochs=10,
):
    """Train the autoencoder with mutual nearest neighbor triplet loss."""
    device, optimizer = args.device, torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=0.0001
    )
    data.x, data.batch, data.cell = (
        data.x.to(device),
        data.batch.to(device),
        data.cell.to(device),
    )
    num_chunks = (data.x.size(0) + chunk_size - 1) // chunk_size
    chunks = create_chunks(data.batch.device, data.batch, num_chunks)
    print(f"Created {num_chunks} chunks.") if verbose else None
    original_embeddings, k_neighbors = data.x.detach().cpu().numpy(), 30
    chunk_neighbor_indices, chunk_neighbor_distances = [], []
    for i, chunk_indices in enumerate(chunks):
        neighbor_indices, neighbor_distances = compute_batch_nearest_neighbors(
            data.batch[chunk_indices],
            embeddings=data.x[chunk_indices],
            k_neighbors=k_neighbors,
        )
        chunk_neighbor_indices.append(
            torch.tensor(neighbor_indices, dtype=torch.long, device=device)
        )
        chunk_neighbor_distances.append(
            torch.tensor(neighbor_distances, dtype=torch.float32, device=device)
        )
    mutual_pairs = None
    train_losses, autoencoder_losses, triplet_losses, neighborhood_losses = (
        [],
        [],
        [],
        [],
    )
    for epoch in range(epochs):
        model.train()
        unique_batches = torch.unique(data.batch).tolist()
        epoch_autoencoder_loss, epoch_triplet_loss, epoch_neighborhood_loss = (
            0.0,
            0.0,
            0.0,
        )
        for i, chunk_indices in enumerate(chunks):
            optimizer.zero_grad()
            chunk_data, chunk_batch_labels = data.x[chunk_indices].to(
                device
            ), data.batch[chunk_indices].to(device)
            latent_embeddings = model.encode(chunk_data, chunk_batch_labels)
            latents_list = [
                latent_embeddings[chunk_batch_labels.squeeze() == b]
                for b in unique_batches
            ]
            recon_x = model.decode(latent_embeddings, chunk_batch_labels)
            autoencoder_loss = model.loss(recon_x, chunk_data)
            neighborhood_loss = neighborhood_preservation_loss_old(
                latent_embeddings,
                chunk_neighbor_indices[i],
                chunk_neighbor_distances[i],
            )
            loss = (
                autoencoder_loss + lamb_neighborhood * neighborhood_loss
                if epoch < pretrain_epochs
                else autoencoder_loss
                + lamb
                * triplet_loss(
                    [latent for latent in latents_list], mutual_pairs, margin=margin
                )
                + lamb_neighborhood * neighborhood_loss
            )
            loss.backward()
            optimizer.step()
            epoch_autoencoder_loss += autoencoder_loss.item()
            epoch_triplet_loss += (
                loss - autoencoder_loss - lamb_neighborhood * neighborhood_loss
            ).item()
            epoch_neighborhood_loss += neighborhood_loss.item()
        train_losses.append(
            (epoch_autoencoder_loss + epoch_triplet_loss + epoch_neighborhood_loss)
            / num_chunks
        )
        autoencoder_losses.append(epoch_autoencoder_loss / num_chunks)
        triplet_losses.append(epoch_triplet_loss / num_chunks)
        neighborhood_losses.append(epoch_neighborhood_loss / num_chunks)
        if epoch % evaluate_interval == 0:
            (
                print(
                    f"Epoch {epoch+1}/{epochs}, Total Loss: {train_losses[-1]:.4f}, AE Loss: {autoencoder_losses[-1]:.4f}, Triplet Loss: {triplet_losses[-1]:.4f}, Neighborhood Loss: {neighborhood_losses[-1]:.4f}, Lambda Triplet: {lamb:.4f}"
                )
                if epoch >= pretrain_epochs
                else print(
                    f"Epoch {epoch+1}/{epochs}, Total Loss: {train_losses[-1]:.4f}, AE Loss: {autoencoder_losses[-1]:.4f}, Neighborhood Loss: {neighborhood_losses[-1]:.4f}"
                )
            )
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_dir = args.checkpoint_dir
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"model_epoch_{epoch+1+checkpoint_start}_{lamb_neighborhood}_0005_margin_1.pth",
            )
            torch.save(model.state_dict(), checkpoint_path)
            print(
                f"Checkpoint saved at epoch {epoch+1+checkpoint_start}: {checkpoint_path}"
            )
    plot_losses(
        epochs, train_losses, autoencoder_losses, triplet_losses, neighborhood_losses
    )
    return model


def plot_losses(
    epochs, train_losses, autoencoder_losses, triplet_losses, neighborhood_losses
):
    """Plot the loss curves after training."""
    epochs_range = range(1, epochs + 1)
    for loss, title, color in zip(
        [train_losses, autoencoder_losses, triplet_losses, neighborhood_losses],
        [
            "Total Loss",
            "Autoencoder Loss",
            "Triplet Loss",
            "Neighborhood Preservation Loss",
        ],
        ["blue", "green", "red", "purple"],
    ):
        plt.figure(figsize=(10, 5))
        plt.plot(epochs_range, loss, label=title, color=color)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"{title} Curve")
        plt.legend()
        plt.show()


def triplet_loss(latents_list, mutual_pairs, margin=1.0):
    """Computes the triplet loss"""
    # If there are no mutual pairs, return a zero loss
    if not mutual_pairs:
        return torch.tensor(0.0, device=latents_list[0].device, requires_grad=True)

    # Initialize lists to store anchor, positive, and negative embeddings
    anchors = []
    positives = []
    negatives = []

    # Loop through each mutual pair
    for pair in mutual_pairs:
        batch_A = pair["batch_A"]
        index_A = pair["index_A"]
        batch_B = pair["batch_B"]
        index_B = pair["index_B"]

        # Get the anchor and positive embeddings
        anchor_embedding = latents_list[batch_A][index_A]
        positive_embedding = latents_list[batch_B][index_B]

        # Get a random negative index from the same batch as the anchor
        negative_index = torch.randint(
            0, latents_list[batch_A].shape[0], (1,), device=latents_list[batch_A].device
        )
        negative_embedding = latents_list[batch_A][negative_index]

        # Store the embeddings
        anchors.append(anchor_embedding)
        positives.append(positive_embedding)
        negatives.append(
            negative_embedding.squeeze(0)
        )  # Remove extra dimension from negative embedding

    # Stack the embeddings into tensors
    anchors = torch.stack(anchors)
    positives = torch.stack(positives)
    negatives = torch.stack(negatives)

    # Compute triplet loss
    triplet_loss_fn = torch.nn.TripletMarginLoss(margin=margin, reduction="mean")
    loss = triplet_loss_fn(anchors, positives, negatives)

    return loss


def neighborhood_preservation_loss_old(
    latent_embeddings, neighbor_indices, original_distances
):
    """Computes the neighborhood preservation loss"""
    # latent embeddings of neighbors
    anchors = latent_embeddings.unsqueeze(1)
    neighbors = latent_embeddings[neighbor_indices]

    # Compute distances in latent space
    latent_distances = torch.norm(anchors - neighbors, dim=2)

    # squared differences between latent and original distances
    distance_diffs = (latent_distances - original_distances) ** 2

    loss = distance_diffs.mean()
    return loss
