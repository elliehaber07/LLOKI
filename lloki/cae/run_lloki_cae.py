import os
import sys

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import torch
from lloki.cae.conditional_autoencoder import ConditionalAutoencoderML
from sklearn.preprocessing import LabelEncoder

from lloki.cae.train import train_autoencoder_mnn_triplet_prechunk

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import matplotlib.pyplot as plt
import numpy as np
import torch

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


def concatenate_anndata(adata_list, slices, subsample_percent=None):
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

    batch_info = torch.tensor(concat_adata.obs['batch'].iloc[subset_indices].values, dtype=torch.long)
    
    label_encoder = LabelEncoder()
    cell_info_encoded = label_encoder.fit_transform(concat_adata.obs['high_level_annotation'].iloc[subset_indices].values)
    cell_info = torch.tensor(cell_info_encoded)

    return Data(
        x=node_features,
        batch=batch_info,
        cell=cell_info
    )


def evaluate_final(data, model, device, args):
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
    
    plt.savefig(f"{args.output_dir}/cae_umap.png")
    return adata


def run_lloki_cae(args):
    torch.cuda.empty_cache()
    device = torch.device("cuda")
    args2=vars(args)
    args2.update({
        'enc_in_channels': 512,  
        'latent_dim': 512,
    })
    batch_dim = 10
    num_batches = 5

    slices = [sc.read(f'{args2["output_dir"]}/{f}') for f in os.listdir(args2["output_dir"]) if 'h5ad' in f]
    for i, s in enumerate(slices):
        s.obs['batch'] = i

    combined_batch = slices
    concatenated_subgraph = concatenate_anndata(combined_batch,slices)

    subset_indices = np.arange(concatenated_subgraph.shape[0])
    data = prepare_data_for_training(concatenated_subgraph, subset_indices)
    data = data.to(device)  # Move the data to the appropriate device
    lamb_neighborhood = args2["npl_num_neighbors"]

    model = ConditionalAutoencoderML(enc_in_channels=args2['enc_in_channels'], 
                            batch_dim=batch_dim, latent_dim=128, hidden_dims=[256, 175], 
                            num_batches=num_batches).to(device)

    model = train_autoencoder_mnn_triplet_prechunk(model, args, data, lr=0.0005, knn=40, epochs=50, pretrain_epochs=0, 
                                        update_interval=1, lamb=0.5, lamb_neighborhood=lamb_neighborhood, chunk_size=4000, checkpoint_interval=1, margin=1)

    evaluate_final(data, model, device, args)