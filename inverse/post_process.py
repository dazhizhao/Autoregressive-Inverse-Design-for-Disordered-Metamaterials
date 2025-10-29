import os
import json
import torch
import numpy as np
import scipy.io as sio
from tqdm import tqdm
from omegaconf import OmegaConf
from main import instantiate_from_config
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


CONFIG_PATH = r"/openbayes/home/configs/config.yaml"
IMG_LIST_PATH = '/openbayes/home/data/imgList.mat'
GROUND_TRUTH_MAT_PATH = "/openbayes/home/data/ind_mat_all.mat"
JSON_SPLIT_PATH = "/openbayes/home/data/inverse_split.json"

OUTPUT_DIR = r"/openbayes/home/inverse/outputs_generated"
VIS_DIR = r"/openbayes/home/inverse/visualizations"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)


def generate_samples(ckpt_path, top_p, temperature):
    """
    Load the specified model and generate samples based on the incoming top_p and temperature.
    """
    config = OmegaConf.load(CONFIG_PATH)
    model = instantiate_from_config(config.model)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict)
    data_module = instantiate_from_config(config.data)
    data_module.setup(stage='predict')
    val_dataloader = data_module.val_dataloader()
    num_samples = len(val_dataloader.dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    grid_h, grid_w = config.model.params.grid_size
    total_steps = grid_h * grid_w
    all_generated_indices = []
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Sample being generated"):
            condition_vectors, _ = batch
            generated_indices = model.sample(
                condition=condition_vectors,
                steps=total_steps,
                batch_size=condition_vectors.shape[0],
                temperature=temperature,
                top_p=top_p
            )
            all_generated_indices.append(generated_indices)
    final_tensor = torch.cat(all_generated_indices, dim=0)
    output_numpy = final_tensor.squeeze(1).cpu().numpy() + 1
    output_numpy_final = output_numpy.transpose(1, 2, 0)
    output_filename_suffix = f"mode_topp_p{top_p}_t{temperature:.2f}"
    output_file_path = os.path.join(OUTPUT_DIR, f"generated_clusters_{num_samples}samples_{output_filename_suffix}.mat")
    sio.savemat(output_file_path, {"generated_indices": output_numpy_final})
    print(f"The file has been saved to: '{output_file_path}'")
    return output_file_path


def reconstruct_single_image(index_matrix, img_list, H, W, cell_H, cell_W):
    """"
    Reconstruction of a complete image from a single index matrix
    """
    final_H = H * cell_H
    final_W = W * cell_W
    reconstructed_image = np.zeros((final_H, final_W), dtype=np.float32)

    for i in range(H):
        for j in range(W):
            cell_type_index = index_matrix[i, j]
            if 1 <= cell_type_index <= len(img_list):
                unit_cell = img_list[cell_type_index - 1].astype(np.float32)
                row_start, row_end = i * cell_H, (i + 1) * cell_H
                col_start, col_end = j * cell_W, (j + 1) * cell_W
                reconstructed_image[row_start:row_end, col_start:col_end] = unit_cell
    
    max_val = reconstructed_image.max()
    if max_val > 0:
        return reconstructed_image / max_val
    return reconstructed_image


def visualize(generated_mat_path, top_p, temperature):
    """
    Load the ground truth and predicted results, reconstruct them separately, and plot the side-by-side comparison.
    Save the comparison plot to a subfolder named after the sampling parameters.
    """
    
    run_suffix = f"p{top_p}_t{temperature:.2f}"
    current_vis_dir = os.path.join(VIS_DIR, run_suffix)
    os.makedirs(current_vis_dir, exist_ok=True)
    print(f"The comparison images for this run will be saved to: '{current_vis_dir}'")
    
    try:
        # 1. Load model prediction results
        pred_struct = sio.loadmat(generated_mat_path)
        pred_var_name = [k for k in pred_struct.keys() if not k.startswith('__')][0]
        predicted_data = pred_struct[pred_var_name].transpose(2, 0, 1) # 转为 (B, H, W)

        # 2. Load ground truth data
        ground_truth_data = sio.loadmat(GROUND_TRUTH_MAT_PATH)['ind_mat_all']

        # 3. Load cell image list
        img_list_struct = sio.loadmat(IMG_LIST_PATH)
        img_list_raw = img_list_struct['imgList']
        img_list = [cell for cell in img_list_raw[0]]

        # 4. Load JSON index file
        with open(JSON_SPLIT_PATH, 'r', encoding='utf-8') as f:
            split_data = json.load(f)['val']

        print('The comparison data for this run was successfully loaded.')
    except Exception as e:
        print(f"Error loading comparison data: {e}")
        return

    H, W = predicted_data.shape[1], predicted_data.shape[2]
    cell_H, cell_W = img_list[0].shape
    sample_idx_counter = 0

    cluster_names = sorted(split_data.keys())
    for cluster_name in cluster_names:
        job_ids_in_cluster = split_data[cluster_name]
        
        cluster_output_dir = os.path.join(current_vis_dir, f"{cluster_name}_Comparisons")
        os.makedirs(cluster_output_dir, exist_ok=True)

        for job_id in tqdm(job_ids_in_cluster, desc=f"Generating comparison plot: {cluster_name}"):
            if sample_idx_counter >= len(predicted_data):
                break

            # 1. Get the model predicted index matrix
            predicted_grid = predicted_data[sample_idx_counter]

            # 2. Get the corresponding ground truth index matrix based on Job ID
            try:
                gt_index = int(job_id.split('-')[1])
                ground_truth_grid = ground_truth_data[:, :, gt_index]
            except (ValueError, IndexError):
                sample_idx_counter += 1
                continue

            # 3. Reconstruct the ground truth and predicted images separately
            gt_reconstructed = reconstruct_single_image(ground_truth_grid, img_list, H, W, cell_H, cell_W)
            pred_reconstructed = reconstruct_single_image(predicted_grid, img_list, H, W, cell_H, cell_W)

            # 4. Create and save the comparison plot using Matplotlib
            fig, axes = plt.subplots(1, 2, figsize=(12, 6))
            
            axes[0].imshow(gt_reconstructed, cmap='gray')
            axes[0].set_title("Ground Truth")
            axes[0].axis('off')
            
            axes[1].imshow(pred_reconstructed, cmap='gray')
            axes[1].set_title("Predicted")
            axes[1].axis('off')
            
            fig.suptitle(f"Comparison for {job_id}", fontsize=16)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            
            output_filepath = os.path.join(cluster_output_dir, f"{job_id}_comparison.png")
            plt.savefig(output_filepath, dpi=200)
            plt.close(fig)

            sample_idx_counter += 1


def run_post_processing(ckpt_path, top_p, temperature):
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"Error: Model checkpoint file does not exist. '{ckpt_path}'")
        return
    generated_mat_file = generate_samples(ckpt_path=ckpt_path, top_p=top_p, temperature=temperature)
    if generated_mat_file and os.path.exists(generated_mat_file):
        visualize(generated_mat_path=generated_mat_file, top_p=top_p, temperature=temperature)
    else:
        print("Error: Failed to generate .mat file, skipping visualization step.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate the index matrix and reconstruct the truth/prediction comparison plot.")
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the trained model checkpoint file (.ckpt)')
    parser.add_argument('--top_p', type=float, default=0.92, help='Top-P sampling probability threshold')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature coefficient')
    args = parser.parse_args()
    run_post_processing(args.ckpt_path, args.top_p, args.temperature)