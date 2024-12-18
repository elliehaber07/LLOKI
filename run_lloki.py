import argparse
import zipfile


from lloki.cae.run_lloki_cae import run_lloki_cae
from lloki.fp.run_lloki_fp import run_lloki_fp
import os
import gdown
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLOKI All")
    parser.add_argument('--data_dir', type=str, default="data/input_slices", help="Directory for input data")
    parser.add_argument('--output_dir', type=str, default="output", help="Directory for saving output")
    parser.add_argument('--model_dir', type=str, default="external/scgpt", help="Directory for model")    
    parser.add_argument('--reference_data_path', type=str, default="data/reference_data/scref_full.h5ad", help="Path to SC Reference adata file")
    parser.add_argument('--checkpoint_dir', type=str, default="checkpoints", help="Directory for cae model checkpoints")    
    parser.add_argument('--k', type=int, default=40, help="K for KNN")
    parser.add_argument('--iter', type=int, default=40, help="Number of iterations")
    parser.add_argument('--alpha', type=float, default=0.5, help="Alpha parameter")
    parser.add_argument('--seed', type=float, default=0, help="Seed")
    parser.add_argument('--device', type=str, default="cuda", help="CUDA device ID (default: 0)")
    parser.add_argument('--npl_num_neighbors', type=int, default=30, help="Number of neighbors for the neighborhood preservation loss")

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.data_dir):
        # Ensure the parent directory of args.data_dir exists
        data_parent_dir = os.path.dirname(args.data_dir)
        os.makedirs(data_parent_dir, exist_ok=True)
        
        # Download the file
        gdown.download(id="1NE6SXmJcEKT4mhVMAO49-Y3KMCNoHOmM", output=os.path.join(data_parent_dir, "h5ads_all.zip"))
        
        # Extract the downloaded zip file
        with zipfile.ZipFile(os.path.join(data_parent_dir, "h5ads_all.zip"), 'r') as zip_ref:
            zip_ref.extractall(data_parent_dir)  # Unzip the contents into the parent directory

    if not os.path.exists(args.model_dir):
        model_parent_dir = os.path.dirname(args.model_dir)
        os.makedirs(model_parent_dir, exist_ok=True)
        gdown.download_folder(url="https://drive.google.com/drive/u/0/folders/1wdVnQWJswC4haO7gOKWZP7AESp5z4kkB",output=model_parent_dir)
        
    run_lloki_fp(args)
    run_lloki_cae(args)
