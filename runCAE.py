import os
import sys
import random
import datetime
import numpy as np
import pandas as pd
import scipy
from scipy.sparse import csr_matrix, vstack
import matplotlib.pyplot as plt
import tqdm
import anndata as ad
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch_geometric.nn.inits import glorot, zeros
from pathlib import Path
from models import scFP_Trainer
from misc.utils import set_seed, set_filename, setup_logger
from argument import parse_args
import scgpt as scg
import importlib
from scipy.sparse import csr_matrix, vstack
from sklearn.preprocessing import LabelEncoder
from anndata import concat as ad_concat
from conditional_autoencoder import ConditionalAutoencoder, ConditionalAutoencoderML
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np
import anndata

high_level_mapping = {
    'Microglia': {
        'RNA_nbclust_clusters_long': ['Microglia', 'Perivascular_macrophages'],
        'Main_molecular_cell_type': ['Microglia'],
        'original_class': ['34 Immune']
    },
    'Astrocytes': {
        'RNA_nbclust_clusters_long': ['Astrocytes_Cortex_Hippocampus', 'Astrocytes_Thalamus_Hypothalamus'],
        'Main_molecular_cell_type': ['Astrocytes'],
        'original_class': ['30 Astro-Epen']
    },
    'Excitatory_neurons': {
        'RNA_nbclust_clusters_long': ['Excitatory_neurons_Layer1_Piriform', 'Excitatory_neurons_Layer2_3', 'Excitatory_neurons_Layer4', 'Excitatory_neurons_Telencephalon', 'Excitatory_neurons_Layer5_6', 'Excitatory_neurons_Hippocampal_CA1', 'Excitatory_neurons_Hippocampal_CA2', 'Excitatory_neurons_Hippocampal_CA3'],
        'Main_molecular_cell_type': ['Telencephalon projecting excitatory neurons'],
        'original_class': ['01 IT-ET Glut', '02 NP-CT-L6b Glut', '03 OB-CR Glut', '13 CNU-HYa Glut', '18 TH Glut', '14 HY Glut', '19 MB Glut', '24 MY Glut', '17 MH-LH Glut', '16 HY MM Glut', '23 P Glut', '15 HY Gnrh1 Glut']
    },
    'Inhibitory_neurons': {
        'RNA_nbclust_clusters_long': ['Inhibitory_neurons_Amygdala', 'Inhibitory_neurons_Habenula_Hypothalamus', 'Inhibitory_neurons_Reticular_nucleus', 'Inhibitory_neurons_Habenula_Thalamus', 'Cck_interneurons', 'Interneurons'],
        'Main_molecular_cell_type': ['Telencephalon inhibitory interneurons'],
        'original_class': ['06 CTX-CGE GABA', '07 CTX-MGE GABA', '05 OB-IMN GABA', '08 CNU-MGE GABA', '09 CNU-LGE GABA', '11 CNU-HYa GABA', '12 HY GABA','20 MB GABA', '27 MY GABA']
    },
    'Oligodendrocytes': {
        'RNA_nbclust_clusters_long': ['Oligodendrocytes_precursor_cells', 'Mature_oligodendrocytes', 'Commited_oligodendrocytes'],
        'Main_molecular_cell_type': ['Oligodendrocyte precursor cells', 'Oligodendrocytes'],
        'original_class': ['31 OPC-Oligo']
    },
    'Vascular_cells': {
        'RNA_nbclust_clusters_long': ['Vascular_leptomeningeal_cells', 'Vascular_smooth_muscle_cells', 'Vascular_endothelial_cells'],
        'Main_molecular_cell_type': ['Vascular and leptomeningeal cells', 'Vascular smooth muscle cells'],
        'original_class': ['33 Vascular']
    },
    'Ependymal_cells': {
        'RNA_nbclust_clusters_long': ['Ependymal_cells'],
        'Main_molecular_cell_type': ['Ependymal cells'],
        'original_class': ['30 Astro-Epen']
    },
    'Other/Unannotated': {
        'RNA_nbclust_clusters_long': ['Neuroblasts', 'Peptidergic_neurons', 'Serotonergic_neurons', 'Choroid_plexus_epithelial_cells', 'Tanycytes'],
        'Main_molecular_cell_type': ['Unannotated'],
        'original_class': []  # Add more classes here if needed
    }
}

# Function to map annotations to high-level category
def map_to_high_level(row, mapping):
    for high_level, annots in mapping.items():
        # Check if each annotation exists in the AnnData object before mapping
        if 'RNA_nbclust_clusters_long' in row.index and row['RNA_nbclust_clusters_long'] in annots['RNA_nbclust_clusters_long']:
            return high_level
        if 'Main_molecular_cell_type' in row.index and row['Main_molecular_cell_type'] in annots['Main_molecular_cell_type']:
            return high_level
        if 'original_class' in row.index and row['original_class'] in annots['original_class']:
        # if 'class' in row.index and row['class'] in annots['original_class']:
            return high_level
    return 'Other/Unannotated'


# def concatenate_anndata(adata_list):
#     concatenated_adata = ad_concat(adata_list, axis=0)
#     return concatenated_adata

def concatenate_anndata(adata_list, subsample_percent=None):
    for ad in slices:
    # Apply the function to create the high-level annotation
        ad.obs['high_level_annotation'] = ad.obs.apply(map_to_high_level, axis=1, mapping=high_level_mapping)
    # Concatenate the AnnData objects along the observation axis (cells)
    concatenated_adata = sc.concat(adata_list, axis=0)
    
    # Subsample if subsample_percent is provided and is between 0 and 100
    if subsample_percent is not None and 0 < subsample_percent < 100:
        # Calculate the number of cells to sample
        n_cells_to_sample = int((subsample_percent / 100) * concatenated_adata.n_obs)
        
        # Randomly select cells to keep
        sampled_indices = np.random.choice(concatenated_adata.n_obs, n_cells_to_sample, replace=False)
        
        # Subset the concatenated_adata to only keep the sampled cells
        concatenated_adata = concatenated_adata[sampled_indices, :].copy()
    
    return concatenated_adata


class Data:
    def __init__(self, x, batch, cell):
        self.x = x
        self.batch = batch
        self.cell = cell

    def to(self, device):
        self.x = self.x.to(device)
        self.batch = self.batch.to(device)
        self.cell = self.cell.to(device)
        return self


def prepare_data_for_training(concat_adata, subset_indices):
    # Convert AnnData object to PyTorch Tensor
    node_features = concat_adata.obsm['X_scGPT'][subset_indices]
    if isinstance(node_features, np.ndarray):
        node_features = torch.FloatTensor(node_features)
    else:  # Assuming it's a sparse matrix
        node_features = torch.FloatTensor(node_features.todense())

    scgpt_info = torch.from_numpy(concat_adata.obsm['X_scGPT'][subset_indices])
    batch_info = torch.tensor(concat_adata.obs['batch'].iloc[subset_indices].values, dtype=torch.long)
    
    label_encoder = LabelEncoder()
    # cell_info_encoded = label_encoder.fit_transform(concat_adata.obs['class'][subset_indices].values)
    cell_info_encoded = label_encoder.fit_transform(concat_adata.obs['high_level_annotation'].iloc[subset_indices].values)
    cell_info = torch.tensor(cell_info_encoded)

    return Data(
        x=node_features,
        batch=batch_info,
        cell=cell_info
    )


def evaluate_final(data, model, device):
    with torch.no_grad():
        x = data.x.to(device)
        batch = data.batch.to(device)
        
        embeddings = model.encode(x, batch).cpu().numpy()  # Get the latent embeddings and move to CPU
        
    adata = ad.AnnData(X=embeddings)
    adata.obs['batch'] = data.batch.cpu().numpy()
    adata.obs['class'] = data.cell.cpu().numpy()

    # Convert batch and class columns to categorical
    adata.obs['batch'] = adata.obs['batch'].astype('category')
    adata.obs['class'] = adata.obs['class'].astype('category')

    sc.pp.neighbors(adata, use_rep='X')  
    sc.tl.leiden(adata, resolution=1.0)
    sc.tl.umap(adata)  
    
    sc.pl.umap(adata, color=['class'])
    sc.pl.umap(adata, color=['batch'])  
    return adata

def find_mutual_nearest_neighbors(embeddings_list, n_neighbors=50):
    # Initialize a list to store mutual nearest neighbor pairs and batch information
    mutual_pairs = []

    # Iterate over all pairs of batches
    for batch_i in range(len(embeddings_list)):
        for batch_j in range(batch_i + 1, len(embeddings_list)):
            embeddings_batch_A = embeddings_list[batch_i]
            embeddings_batch_B = embeddings_list[batch_j]

            # Find neighbors from batch A to batch B
            knn_A_to_B = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine').fit(embeddings_batch_B)
            distances_A_to_B, indices_A_to_B = knn_A_to_B.kneighbors(embeddings_batch_A)

            # Find neighbors from batch B to batch A
            knn_B_to_A = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine').fit(embeddings_batch_A)
            distances_B_to_A, indices_B_to_A = knn_B_to_A.kneighbors(embeddings_batch_B)

            # Find mutual nearest neighbors between the two batches
            for i, neighbors_in_B in enumerate(indices_A_to_B):
                for neighbor_in_B in neighbors_in_B:
                    if i in indices_B_to_A[neighbor_in_B]:
                        # Store the pair along with their respective batches
                        mutual_pairs.append({
                            'batch_A': batch_i,
                            'index_A': i,
                            'batch_B': batch_j,
                            'index_B': neighbor_in_B
                        })
                        break  # Mutual pair found, stop looking for further neighbors

    print(f"Number of mutual pairs found: {len(mutual_pairs)}")
    return mutual_pairs

def triplet_loss(latents_list, mutual_pairs, margin=1.0):
    # If there are no mutual pairs, return a zero loss
    if not mutual_pairs:
        return torch.tensor(0.0, device=latents_list[0].device, requires_grad=True)
    
    # Initialize lists to store anchor, positive, and negative embeddings
    anchors = []
    positives = []
    negatives = []
    
    # Loop through each mutual pair
    for pair in mutual_pairs:
        batch_A = pair['batch_A']
        index_A = pair['index_A']
        batch_B = pair['batch_B']
        index_B = pair['index_B']
        
        # Get the anchor and positive embeddings
        anchor_embedding = latents_list[batch_A][index_A]
        positive_embedding = latents_list[batch_B][index_B]

        # Get a random negative index from the same batch as the anchor
        negative_index = torch.randint(0, latents_list[batch_A].shape[0], (1,), device=latents_list[batch_A].device)
        negative_embedding = latents_list[batch_A][negative_index]
        
        # Store the embeddings
        anchors.append(anchor_embedding)
        positives.append(positive_embedding)
        negatives.append(negative_embedding.squeeze(0))  # Remove extra dimension from negative embedding
    
    # Stack the embeddings into tensors
    anchors = torch.stack(anchors)
    positives = torch.stack(positives)
    negatives = torch.stack(negatives)
    
    # Compute triplet loss
    triplet_loss_fn = torch.nn.TripletMarginLoss(margin=margin, reduction='mean')
    loss = triplet_loss_fn(anchors, positives, negatives)
    
    return loss

'''
# Function to compute neighborhood preservation loss
def neighborhood_preservation_loss(latent_embeddings, neighbor_indices, original_distances, chunk_indices):
    """
    Compute the neighborhood preservation loss for the current chunk.
    :param latent_embeddings: Latent embeddings for the current chunk.
    :param neighbor_indices: Neighbor indices (global indices).
    :param original_distances: Original distances (from the full dataset).
    :param chunk_indices: The global indices of the current chunk.
    :return: Neighborhood preservation loss.
    """
    print(f'chunk_indices: {chunk_indices}')
    # Create a mapping from global to chunk-specific indices
    chunk_indices = chunk_indices.cpu().numpy()
    adjusted_neighbor_indices = []
    
    # Loop over each row in neighbor_indices
    for row in neighbor_indices:
        adjusted_row = []
        for global_idx in row:
            if int(global_idx) in chunk_indices:
                # Get the position of the global_idx in chunk_indices
                # print('its here')
                chunk_idx_position = np.where(chunk_indices == int(global_idx))[0]
                if len(chunk_idx_position) > 0:
                    adjusted_row.append(chunk_idx_position[0])  # Add the relative index in the chunk
                else:
                    adjusted_row.append(-1)  # Handle the edge case where it's not found
            else:
                adjusted_row.append(-1)  # Invalid neighbor (not in the chunk)
        adjusted_neighbor_indices.append(adjusted_row)

    # Convert adjusted indices to tensor
    adjusted_neighbor_indices = torch.tensor(adjusted_neighbor_indices, dtype=torch.long, device=latent_embeddings.device)
    print(f'adjusted_neighbor_indices: {adjusted_neighbor_indices}')
    # Filter out invalid neighbors (-1)
    valid_mask = adjusted_neighbor_indices != -1
    adjusted_neighbor_indices = adjusted_neighbor_indices[valid_mask].view(latent_embeddings.size(0), -1)
    print(f'adjusted_neighbor_indices: {adjusted_neighbor_indices}')
    original_distances = original_distances[valid_mask].view(latent_embeddings.size(0), -1)
    print(f'original_distances: {original_distances}')

    # Get neighbors' latent embeddings using the adjusted indices
    neighbors = latent_embeddings[adjusted_neighbor_indices]

    # Compute distances in latent space
    anchors = latent_embeddings.unsqueeze(1)  # [batch_size, 1, embedding_size]
    latent_distances = torch.norm(anchors - neighbors, dim=2)  # [batch_size, num_neighbors]
    print(latent_distances)

    # Compute squared differences between latent and original distances
    distance_diffs = (latent_distances - original_distances) ** 2
    print(f'distance diffs: {distance_diffs}')

    # Compute and return the loss
    loss = distance_diffs.mean()
    return loss
'''
def neighborhood_preservation_loss(latent_embeddings, neighbor_indices, original_distances, chunk_indices, n_neighbors=30):
    """
    Compute the neighborhood preservation loss for the current chunk, ensuring we always have a fixed number of valid neighbors.
    :param latent_embeddings: Latent embeddings for the current chunk.
    :param neighbor_indices: Neighbor indices (global indices).
    :param original_distances: Original distances (from the full dataset).
    :param chunk_indices: The global indices of the current chunk.
    :param num_neighbors: Number of neighbors to keep after filtering.
    :return: Neighborhood preservation loss.
    """
    # Ensure chunk_indices is a numpy array
    chunk_indices = chunk_indices.cpu().numpy()
    '''
    # Initialize the adjusted neighbor indices list
    adjusted_neighbor_indices = []
    adjusted_original_distances = []  # Keep track of corresponding original distances for valid neighbors
    
    # Loop over each row in neighbor_indices
    for i, row in enumerate(neighbor_indices):  # Take extra neighbors initially
        adjusted_row = []
        adjusted_distances_row = []
        for j, global_idx in enumerate(row):
            if int(global_idx) in chunk_indices:
                # Get the position of the global_idx in chunk_indices
                chunk_idx_position = np.where(chunk_indices == int(global_idx))[0]
                if len(chunk_idx_position) > 0:
                    adjusted_row.append(chunk_idx_position[0])  # Add the relative index in the chunk
                    adjusted_distances_row.append(original_distances[i][j])  # Add the corresponding distance
            else:
                adjusted_row.append(-1)  # Invalid neighbor (not in the chunk)
                adjusted_distances_row.append(1000)  # Mark invalid distances as infinity

        # Append valid rows
        adjusted_neighbor_indices.append(adjusted_row)
        adjusted_original_distances.append(adjusted_distances_row)
    '''
    global_to_chunk_map = {int(global_idx): i for i, global_idx in enumerate(chunk_indices)}
    
    # Initialize placeholder tensors for adjusted neighbors and distances
    adjusted_neighbor_indices = np.full(neighbor_indices.shape, -1)  # Initialize with -1 (invalid)
    adjusted_original_distances = np.full(neighbor_indices.shape, 1000.0)  # Initialize with large value (invalid distances)

    # Vectorize the process of finding valid neighbors
    for i in range(neighbor_indices.shape[0]):
        global_indices = neighbor_indices[i].cpu().numpy().astype(int)
        valid_mask = np.isin(global_indices, chunk_indices)

        original_distances_cpu = original_distances[i].cpu().numpy()

        # Map global indices to chunk-specific indices
        chunk_idx_positions = [global_to_chunk_map[global_idx] if global_idx in global_to_chunk_map else -1 for global_idx in global_indices]
        
        # Update valid neighbors
        adjusted_neighbor_indices[i, valid_mask] = np.array(chunk_idx_positions)[valid_mask]
        adjusted_original_distances[i, valid_mask] = original_distances_cpu[valid_mask]
    # print('built adjusted_neighbor_indices')
    # Convert adjusted indices and distances to tensors
    adjusted_neighbor_indices = torch.tensor(adjusted_neighbor_indices, dtype=torch.long, device=latent_embeddings.device)
    adjusted_original_distances = torch.tensor(adjusted_original_distances, dtype=torch.float32, device=latent_embeddings.device)
    # print(adjusted_original_distances)
    # Sort by distance and select the closest valid neighbors (up to n_neighbors)
    sorted_indices = torch.argsort(adjusted_original_distances, dim=1)
    closest_indices = sorted_indices[:, :n_neighbors]

    # Gather the closest valid neighbors and distances
    valid_neighbor_indices = torch.gather(adjusted_neighbor_indices, 1, closest_indices)
    valid_original_distances = torch.gather(adjusted_original_distances, 1, closest_indices)
    # print(valid_original_distances)

    # Get neighbors' latent embeddings using the valid neighbor indices
    valid_neighbors = latent_embeddings[valid_neighbor_indices]

    # Compute distances in latent space
    anchors = latent_embeddings.unsqueeze(1)  # [batch_size, 1, embedding_size]
    # print('finding latent_distances')
    latent_distances = torch.norm(anchors - valid_neighbors, dim=2)  # [batch_size, n_neighbors]

    # Compute squared differences between latent and original distances
    distance_diffs = (latent_distances - valid_original_distances) ** 2

    # Compute and return the loss
    loss = distance_diffs.mean()
    return loss

def neighborhood_preservation_loss_old(latent_embeddings, neighbor_indices, original_distances):
    # latent embeddings of neighbors
    anchors = latent_embeddings.unsqueeze(1)  
    neighbors = latent_embeddings[neighbor_indices]  

    # Compute distances in latent space
    latent_distances = torch.norm(anchors - neighbors, dim=2)  

    # squared differences between latent and original distances
    distance_diffs = (latent_distances - original_distances) ** 2 

    loss = distance_diffs.mean()
    return loss
import random
'''
def create_chunks(latent_embeddings, batch_labels, num_chunks):
    """
    Split data into chunks while ensuring each chunk has balanced samples from all batches.
    :param latent_embeddings: Tensor of latent embeddings for all samples.
    :param batch_labels: Tensor of batch labels corresponding to each sample in latent_embeddings.
    :param num_chunks: Number of chunks to create.
    :return: List of tensors where each tensor corresponds to a chunk of balanced latent embeddings.
    """
    unique_batches = torch.unique(batch_labels).tolist()
    batch_indices = {b: torch.where(batch_labels == b)[0] for b in unique_batches}

    # Shuffle the indices within each batch
    for indices in batch_indices.values():
        indices = indices[torch.randperm(indices.size(0))]

    # Create chunks with balanced representation of each batch
    chunks = []
    for _ in range(num_chunks):
        chunk_indices = []
        for b in unique_batches:
            # Split the indices of each batch evenly among the chunks
            chunk_size_per_batch = batch_indices[b].size(0) // num_chunks
            chunk_indices.append(batch_indices[b][:chunk_size_per_batch])
            batch_indices[b] = batch_indices[b][chunk_size_per_batch:]

        # Concatenate the indices from different batches to form the final chunk
        chunk_indices = torch.cat(chunk_indices)
        chunks.append(chunk_indices)

    return chunks
'''

def create_chunks(device, batch_labels, num_chunks):
    """
    Split data into chunks while ensuring each chunk has balanced samples from all batches.
    :param latent_embeddings: Tensor of latent embeddings for all samples.
    :param batch_labels: Tensor of batch labels corresponding to each sample in latent_embeddings.
    :param num_chunks: Number of chunks to create.
    :return: List of tensors where each tensor corresponds to a chunk of balanced latent embeddings.
    """
    unique_batches = torch.unique(batch_labels).tolist()
    batch_indices = {b: torch.where(batch_labels == b)[0] for b in unique_batches}

    # Shuffle the indices within each batch
    for b in unique_batches:
        batch_indices[b] = batch_indices[b][torch.randperm(batch_indices[b].size(0))]

    # Initialize chunks list
    chunks = [torch.empty(0, dtype=torch.long, device=device) for _ in range(num_chunks)]

    # Distribute samples from each batch evenly across chunks
    for b in unique_batches:
        num_samples_per_chunk = batch_indices[b].size(0) // num_chunks
        remainder = batch_indices[b].size(0) % num_chunks

        # Evenly distribute samples across chunks
        start_idx = 0
        for i in range(num_chunks):
            end_idx = start_idx + num_samples_per_chunk
            chunks[i] = torch.cat((chunks[i], batch_indices[b][start_idx:end_idx]))
            start_idx = end_idx

        # Distribute remaining samples randomly across chunks
        if remainder > 0:
            remainder_indices = batch_indices[b][start_idx:]
            random_chunk_indices = torch.randperm(remainder)
            for i, remainder_idx in enumerate(random_chunk_indices):
                chunks[i] = torch.cat((chunks[i], remainder_indices[remainder_idx:remainder_idx + 1]))

    return chunks


def monitor_gpu_memory():
    allocated_memory = torch.cuda.memory_allocated(device)
    max_allocated_memory = torch.cuda.max_memory_allocated(device)
    print(f"Currently allocated memory: {allocated_memory / (1024 ** 3):.2f} GB")
    print(f"Max memory allocated so far: {max_allocated_memory / (1024 ** 3):.2f} GB")

def compute_batch_nearest_neighbors(data, embeddings, k_neighbors=30):
    """
    Compute nearest neighbors for each batch independently, using `data.batch` to find batch information.
    :param data: The data object containing batch information.
    :param embeddings: The embeddings from the data (e.g., data.x or latent embeddings).
    :param k_neighbors: Number of neighbors to compute for each batch.
    :return: A tuple of neighbor indices and distances tensors corresponding to the entire dataset.
    """
    # Initialize lists to hold neighbors and distances
    all_neighbor_indices = []
    all_neighbor_distances = []

    # Loop over each batch
    for batch in torch.unique(data.batch).tolist():
        # Get embeddings and indices for the current batch
        batch_indices = torch.where(data.batch == batch)[0].cpu().numpy()
        batch_embeddings = embeddings[batch_indices].cpu().numpy()

        # Fit NearestNeighbors on the batch-specific embeddings
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='auto', metric='euclidean').fit(batch_embeddings)

        # Find neighbors within the batch
        distances, indices = nbrs.kneighbors(batch_embeddings)

        # Exclude the self-neighbor at index 0 and store results
        all_neighbor_indices.append(batch_indices[indices[:, 1:]])
        all_neighbor_distances.append(distances[:, 1:])

    # Concatenate the results to match the original dataset shape
    neighbor_indices = np.concatenate(all_neighbor_indices, axis=0)
    neighbor_distances = np.concatenate(all_neighbor_distances, axis=0)

    return neighbor_indices, neighbor_distances

def compute_batch_nearest_neighbors2(batchdata, embeddings, k_neighbors=30):
    """
    Compute nearest neighbors for each batch independently, using `data.batch` to find batch information.
    :param data: The data object containing batch information.
    :param embeddings: The embeddings from the data (e.g., data.x or latent embeddings).
    :param k_neighbors: Number of neighbors to compute for each batch.
    :return: A tuple of neighbor indices and distances tensors corresponding to the entire dataset.
    """
    # Initialize lists to hold neighbors and distances
    all_neighbor_indices = []
    all_neighbor_distances = []

    # Loop over each batch
    for batch in torch.unique(batchdata).tolist():
        # Get embeddings and indices for the current batch
        batch_indices = torch.where(batchdata == batch)[0].cpu().numpy()
        batch_embeddings = embeddings[batch_indices].cpu().numpy()

        # Fit NearestNeighbors on the batch-specific embeddings
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='auto', metric='euclidean').fit(batch_embeddings)

        # Find neighbors within the batch
        distances, indices = nbrs.kneighbors(batch_embeddings)

        # Exclude the self-neighbor at index 0 and store results
        all_neighbor_indices.append(batch_indices[indices[:, 1:]])
        all_neighbor_distances.append(distances[:, 1:])

    # Concatenate the results to match the original dataset shape
    neighbor_indices = np.concatenate(all_neighbor_indices, axis=0)
    neighbor_distances = np.concatenate(all_neighbor_distances, axis=0)

    return neighbor_indices, neighbor_distances


def train_autoencoder_mnn_triplet(
    model,
    args,
    data,
    lr,
    epochs,
    evaluate_interval=10,
    lamb=0.5,  
    lamb_neighborhood=15.0, 
    pretrain_epochs=50,
    margin=1.0,
    knn=100,
    update_interval=15,
    chunk_size=512,  # Set chunk size based on memory constraints
    verbose=False,
    checkpoint_interval=25,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    
    # Move data to the device
    data.x = data.x.to(device)
    data.batch = data.batch.to(device)
    data.cell = data.cell.to(device)
    
    # Prepare original embeddings for neighborhood preservation loss
    original_embeddings = data.x.detach().cpu().numpy()

    # Number of neighbors for neighborhood preservation
    k_neighbors = 120
    k_neighbors_small = 30

    if verbose == True:
        print('Starting finding neighbors for neighborhood loss')
    neighbor_indices, neighbor_distances = compute_batch_nearest_neighbors(data, embeddings=data.x, k_neighbors=k_neighbors) 
    if verbose == True:
        print('Done finding neighbors for neighborhood loss')
    
    # Convert neighbor data to tensors
    neighbor_indices_tensor = torch.tensor(neighbor_indices, dtype=torch.long, device=device)
    original_distances_tensor = torch.tensor(neighbor_distances, dtype=torch.float32, device=device)

    # nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='auto', metric='euclidean').fit(original_embeddings)
    # distances, indices = nbrs.kneighbors(original_embeddings)

    # # Exclude self 
    # neighbor_indices = indices[:, 1:]
    # neighbor_distances = distances[:, 1:]  

    # neighbor_indices_tensor = torch.tensor(neighbor_indices, dtype=torch.long, device=device)
    # original_distances_tensor = torch.tensor(neighbor_distances, dtype=torch.float32, device=device)

    # Track the loss 
    train_losses = []
    autoencoder_losses = []
    triplet_losses = []
    neighborhood_losses = []

    mutual_pairs = None  # Initialize mutual_pairs to None
    
    # Total epochs after pretraining
    total_epochs_post_pretrain = epochs - pretrain_epochs

    for epoch in range(epochs):
        model.train()

        # Split latent embeddings by batch
        unique_batches = torch.unique(data.batch).tolist()

        # Define the number of chunks
        num_chunks = (data.x.size(0) + chunk_size - 1) // chunk_size

        # Recalculate chunks at the beginning of each epoch
        chunks = create_chunks(data.batch.device, data.batch, num_chunks)
        if verbose == True:
            print(f'Created {num_chunks} chunks of data. Processing now')
        # monitor_gpu_memory()

        # Initialize epoch losses
        epoch_autoencoder_loss = 0.0
        epoch_triplet_loss = 0.0
        epoch_neighborhood_loss = 0.0

        # Process each chunk
        for chunk_indices in chunks:
            if verbose == True:
                print('starting next chunk')
            optimizer.zero_grad()
            chunk_data = data.x[chunk_indices].to(device)
            chunk_batch_labels = data.batch[chunk_indices].to(device)
            latent_embeddings = model.encode(chunk_data, chunk_batch_labels)
            
            chunk_embeddings = latent_embeddings

            # Split chunked latent embeddings by batch
            latents_list = [chunk_embeddings[chunk_batch_labels.squeeze() == b] for b in unique_batches]

            # Reconstruct from latent embeddings
            recon_x = model.decode(chunk_embeddings, chunk_batch_labels)
            autoencoder_loss = model.loss(recon_x, chunk_data)
            if verbose == True:
                print('got autoencoder loss for this chunk')

            # Compute neighborhood preservation loss for the chunk
            # neighborhood_loss = neighborhood_preservation_loss(chunk_embeddings, neighbor_indices_tensor[chunk_indices], original_distances_tensor[chunk_indices])
            neighborhood_loss = neighborhood_preservation_loss(
                chunk_embeddings, 
                neighbor_indices_tensor[chunk_indices], 
                original_distances_tensor[chunk_indices], 
                chunk_indices,
                n_neighbors=k_neighbors_small,
            )
            if verbose == True:
                print('got neighborhood loss for this chunk')
            if epoch < pretrain_epochs:
                # Only use autoencoder and neighborhood loss during pre-training
                loss = autoencoder_loss + lamb_neighborhood * neighborhood_loss
                triplet_loss_value = torch.tensor(0.0, device=device, requires_grad=True)
                lamb_current = 0.0  # No triplet loss during pre-training
            else:
                # Gradually increase lamb over epochs
                lamb_current = lamb * ((epoch - pretrain_epochs + 1) / total_epochs_post_pretrain)
                lamb_current = min(lamb_current, lamb)

                # Recompute mutual pairs every update_interval epochs or at first run
                if (epoch - pretrain_epochs) % update_interval == 0 or mutual_pairs is None:
                    print(f"Recomputing mutual pairs at epoch {epoch}")

                    # Find mutual nearest neighbors across all batches
                    latents_np_list = [latent.detach().cpu().numpy() for latent in latents_list]
                    mutual_pairs = find_mutual_nearest_neighbors(latents_np_list, n_neighbors=knn)

                # Compute triplet loss if mutual pairs are available
                if not mutual_pairs:
                    print("No mutual pairs found.")
                    triplet_loss_value = torch.tensor(0.0, device=device, requires_grad=True)
                else:
                    # triplet_loss_value = triplet_loss([latent.detach() for latent in latents_list], mutual_pairs, margin=margin)
                    triplet_loss_value = triplet_loss([latent for latent in latents_list], mutual_pairs, margin=margin)
                    if verbose == True:
                        print('got triplet loss for this chunk')

                # Combine the losses
                loss = autoencoder_loss + lamb_current * triplet_loss_value + lamb_neighborhood * neighborhood_loss
                if verbose == True:
                    print(f'loss for this chunk: {loss}')
                    print(f'losses for this chunk: ae:{autoencoder_loss}, nn: {neighborhood_loss}, trip: {triplet_loss_value}')

            # Accumulate losses for the epoch
            epoch_autoencoder_loss += autoencoder_loss.item()
            epoch_triplet_loss += triplet_loss_value.item()
            epoch_neighborhood_loss += neighborhood_loss.item()

            # Backpropagate and update the model parameters
            loss.backward()
            if verbose == True:
                print('done back prop')
            optimizer.step()
            # monitor_gpu_memory()

        # Average losses over chunks
        train_losses.append((epoch_autoencoder_loss + epoch_triplet_loss + epoch_neighborhood_loss) / num_chunks)
        autoencoder_losses.append(epoch_autoencoder_loss / num_chunks)
        triplet_losses.append(epoch_triplet_loss / num_chunks)
        neighborhood_losses.append(epoch_neighborhood_loss / num_chunks)

        # Print loss at intervals
        if epoch % evaluate_interval == 0:
            if epoch >= pretrain_epochs:
                print(f'Epoch {epoch+1}/{epochs}, Total Loss: {train_losses[-1]:.4f}, AE Loss: {autoencoder_losses[-1]:.4f}, '
                      f'Triplet Loss: {triplet_losses[-1]:.4f}, Neighborhood Loss: {neighborhood_losses[-1]:.4f}, '
                      f'Lambda Triplet: {lamb_current:.4f}')
            else:
                print(f'Epoch {epoch+1}/{epochs}, Total Loss: {train_losses[-1]:.4f}, AE Loss: {autoencoder_losses[-1]:.4f}, '
                      f'Neighborhood Loss: {neighborhood_losses[-1]:.4f}')
            
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_dir = '/work/magroup/skrieger/LLOKI/checkpoints'
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1}.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Checkpoint saved at epoch {epoch+1}: {checkpoint_path}')
    # Plot the loss curves after training
    epochs_range = range(1, epochs + 1)

    # Plot Total Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_losses, label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Total Loss Curve')
    plt.legend()
    plt.show()

    # Plot Autoencoder Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, autoencoder_losses, label='Autoencoder Loss', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Autoencoder Loss Curve')
    plt.legend()
    plt.show()

    # Plot Triplet Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, triplet_losses, label='Triplet Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Triplet Loss Curve')
    plt.legend()
    plt.show()

    # Plot Neighborhood Preservation Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, neighborhood_losses, label='Neighborhood Loss', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Neighborhood Preservation Loss Curve')
    plt.legend()
    plt.show()
    
    return model


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
    chunk_size=512,  # Set chunk size based on memory constraints
    verbose=False,
    checkpoint_interval=25,
    checkpoint_start=0,
    ramp_up_epochs=10,
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    
    # Move data to the device
    data.x = data.x.to(device)
    data.batch = data.batch.to(device)
    data.cell = data.cell.to(device)

    # Define the number of chunks
    num_chunks = (data.x.size(0) + chunk_size - 1) // chunk_size

    # Recalculate chunks at the beginning of each epoch
    chunks = create_chunks(data.batch.device, data.batch, num_chunks)
    # if verbose == True:
    print(f'Created {num_chunks} chunks of data. Processing now')
    
    # Prepare original embeddings for neighborhood preservation loss
    original_embeddings = data.x.detach().cpu().numpy()

    # Number of neighbors for neighborhood preservation
    k_neighbors = 30

    if verbose == True:
        print('Starting finding neighbors for neighborhood loss')
    chunk_neighbor_indices = []
    chunk_neighbor_distances = []
    for i, chunk_indices in enumerate(chunks):
        neighbor_indices, neighbor_distances = compute_batch_nearest_neighbors2(
                                        data.batch[chunk_indices], 
                                        embeddings=data.x[chunk_indices], 
                                        k_neighbors=k_neighbors) 
        neighbor_indices_tensor = torch.tensor(neighbor_indices, dtype=torch.long, device=device)
        original_distances_tensor = torch.tensor(neighbor_distances, dtype=torch.float32, device=device)
        chunk_neighbor_indices.append(neighbor_indices_tensor)
        chunk_neighbor_distances.append(original_distances_tensor)
    if verbose == True:
        print('Done finding neighbors for neighborhood loss')
    
    # Convert neighbor data to tensors
    

    # nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, algorithm='auto', metric='euclidean').fit(original_embeddings)
    # distances, indices = nbrs.kneighbors(original_embeddings)

    # # Exclude self 
    # neighbor_indices = indices[:, 1:]
    # neighbor_distances = distances[:, 1:]  

    # neighbor_indices_tensor = torch.tensor(neighbor_indices, dtype=torch.long, device=device)
    # original_distances_tensor = torch.tensor(neighbor_distances, dtype=torch.float32, device=device)

    # Track the loss 
    train_losses = []
    autoencoder_losses = []
    triplet_losses = []
    neighborhood_losses = []

    mutual_pairs = None  # Initialize mutual_pairs to None
    
    # Total epochs after pretraining
    total_epochs_post_pretrain = epochs - pretrain_epochs

    for epoch in range(epochs):
        model.train()

        # Split latent embeddings by batch
        unique_batches = torch.unique(data.batch).tolist()

        
        # monitor_gpu_memory()

        # Initialize epoch losses
        epoch_autoencoder_loss = 0.0
        epoch_triplet_loss = 0.0
        epoch_neighborhood_loss = 0.0

        # Process each chunk
        for i,chunk_indices in enumerate(chunks):
            if verbose == True:
                print('starting next chunk')
            optimizer.zero_grad()
            chunk_data = data.x[chunk_indices].to(device)
            chunk_batch_labels = data.batch[chunk_indices].to(device)
            latent_embeddings = model.encode(chunk_data, chunk_batch_labels)
            
            chunk_embeddings = latent_embeddings

            # Split chunked latent embeddings by batch
            latents_list = [chunk_embeddings[chunk_batch_labels.squeeze() == b] for b in unique_batches]

            # Reconstruct from latent embeddings
            recon_x = model.decode(chunk_embeddings, chunk_batch_labels)
            autoencoder_loss = model.loss(recon_x, chunk_data)
            if verbose == True:
                print('got autoencoder loss for this chunk')

            # Compute neighborhood preservation loss for the chunk
            # neighborhood_loss = neighborhood_preservation_loss(chunk_embeddings, neighbor_indices_tensor[chunk_indices], original_distances_tensor[chunk_indices])
            neighborhood_loss = neighborhood_preservation_loss_old(
                chunk_embeddings, 
                chunk_neighbor_indices[i], 
                chunk_neighbor_distances[i], 
            )
            if verbose == True:
                print('got neighborhood loss for this chunk')
            if epoch < pretrain_epochs:
                # Only use autoencoder and neighborhood loss during pre-training
                loss = autoencoder_loss + lamb_neighborhood * neighborhood_loss
                triplet_loss_value = torch.tensor(0.0, device=device, requires_grad=True)
                lamb_current = 0.0  # No triplet loss during pre-training
            else:
                # Gradually increase lamb over epochs
                lamb_current = lamb * ((epoch - pretrain_epochs + 1) / ramp_up_epochs)
                lamb_current = min(lamb_current, lamb)

                # Recompute mutual pairs every update_interval epochs or at first run
                if (epoch - pretrain_epochs) % update_interval == 0 or mutual_pairs is None:
                    print(f"Recomputing mutual pairs at epoch {epoch}")

                    # Find mutual nearest neighbors across all batches
                    latents_np_list = [latent.detach().cpu().numpy() for latent in latents_list]
                    mutual_pairs = find_mutual_nearest_neighbors(latents_np_list, n_neighbors=knn)

                # Compute triplet loss if mutual pairs are available
                if not mutual_pairs:
                    print("No mutual pairs found.")
                    triplet_loss_value = torch.tensor(0.0, device=device, requires_grad=True)
                else:
                    # triplet_loss_value = triplet_loss([latent.detach() for latent in latents_list], mutual_pairs, margin=margin)
                    triplet_loss_value = triplet_loss([latent for latent in latents_list], mutual_pairs, margin=margin)
                    if verbose == True:
                        print('got triplet loss for this chunk')

                # Combine the losses
                loss = autoencoder_loss + lamb_current * triplet_loss_value + lamb_neighborhood * neighborhood_loss
                if verbose == True:
                    print(f'loss for this chunk: {loss}')
                    print(f'losses for this chunk: ae:{autoencoder_loss}, nn: {neighborhood_loss}, trip: {triplet_loss_value}')

            # Accumulate losses for the epoch
            epoch_autoencoder_loss += autoencoder_loss.item()
            epoch_triplet_loss += triplet_loss_value.item()
            epoch_neighborhood_loss += neighborhood_loss.item()

            # Backpropagate and update the model parameters
            loss.backward()
            if verbose == True:
                print('done back prop')
            optimizer.step()
            # monitor_gpu_memory()

        # Average losses over chunks
        train_losses.append((epoch_autoencoder_loss + epoch_triplet_loss + epoch_neighborhood_loss) / num_chunks)
        autoencoder_losses.append(epoch_autoencoder_loss / num_chunks)
        triplet_losses.append(epoch_triplet_loss / num_chunks)
        neighborhood_losses.append(epoch_neighborhood_loss / num_chunks)

        # Print loss at intervals
        if epoch % evaluate_interval == 0:
            if epoch >= pretrain_epochs:
                print(f'Epoch {epoch+1}/{epochs}, Total Loss: {train_losses[-1]:.4f}, AE Loss: {autoencoder_losses[-1]:.4f}, '
                      f'Triplet Loss: {triplet_losses[-1]:.4f}, Neighborhood Loss: {neighborhood_losses[-1]:.4f}, '
                      f'Lambda Triplet: {lamb_current:.4f}')
            else:
                print(f'Epoch {epoch+1}/{epochs}, Total Loss: {train_losses[-1]:.4f}, AE Loss: {autoencoder_losses[-1]:.4f}, '
                      f'Neighborhood Loss: {neighborhood_losses[-1]:.4f}')
            
        if (epoch + 1) % checkpoint_interval == 0:
            checkpoint_dir = '/work/magroup/skrieger/LLOKI/checkpoints'
            checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{epoch+1+checkpoint_start}_{lamb_neighborhood}_0005_margin_1.pth')
            torch.save(model.state_dict(), checkpoint_path)
            print(f'Checkpoint saved at epoch {epoch+1+checkpoint_start}: {checkpoint_path}')
    # Plot the loss curves after training
    epochs_range = range(1, epochs + 1)

    # Plot Total Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, train_losses, label='Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Total Loss Curve')
    plt.legend()
    plt.show()

    # Plot Autoencoder Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, autoencoder_losses, label='Autoencoder Loss', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Autoencoder Loss Curve')
    plt.legend()
    plt.show()

    # Plot Triplet Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, triplet_losses, label='Triplet Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Triplet Loss Curve')
    plt.legend()
    plt.show()

    # Plot Neighborhood Preservation Loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs_range, neighborhood_losses, label='Neighborhood Loss', color='purple')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Neighborhood Preservation Loss Curve')
    plt.legend()
    plt.show()
    
    return model

torch.cuda.empty_cache()
device = torch.device("cuda")

args = {
    'enc_in_channels': 512,  
    'latent_dim': 512  
}
batch_dim = 10
num_batches = 5
# model = ConditionalAutoencoder(enc_in_channels=args['enc_in_channels'], batch_dim=batch_dim, latent_dim=args['latent_dim']).to(device)

slices = [sc.read(f'../LLOKI_datasets/Five-slices_processed/spatial_weighting/{f}') for f in os.listdir('../LLOKI_datasets/Five-slices_processed/spatial_weighting') if 'h5ad' in f]
for i, s in enumerate(slices):
    s.obs['batch'] = i

combined_batch = slices
# concatenated_subgraph = concatenate_anndata(combined_batch, subsample_percent=10)
concatenated_subgraph = concatenate_anndata(combined_batch)

subset_indices = np.arange(concatenated_subgraph.shape[0])
data = prepare_data_for_training(concatenated_subgraph, subset_indices)
# data.batch = data.batch.unsqueeze(1).to(device)
data = data.to(device)  # Move the data to the appropriate device
lamb_neighborhood = int(sys.argv[1])

model = ConditionalAutoencoderML(enc_in_channels=args['enc_in_channels'], 
                         batch_dim=batch_dim, latent_dim=128, hidden_dims=[256, 175], 
                         num_batches=num_batches).to(device)

model = train_autoencoder_mnn_triplet_prechunk(model, args, data, lr=0.0005, knn=40, epochs=50, pretrain_epochs=0, 
                                      update_interval=1, lamb=0.5, lamb_neighborhood=lamb_neighborhood, chunk_size=4000, checkpoint_interval=1, margin=1)
