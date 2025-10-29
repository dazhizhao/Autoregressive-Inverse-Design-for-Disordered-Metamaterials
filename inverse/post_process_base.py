import os
import json
import torch
import numpy as np
import scipy.io as sio
from tqdm import tqdm
from omegaconf import OmegaConf
from main import instantiate_from_config
import argparse
import pandas as pd
import re
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


CONFIG_PATH = r"/openbayes/home/configs/config.yaml"
IMG_LIST_PATH = '/openbayes/home/data/imgList.mat'
ORIGINAL_EXCEL_PATH = "/openbayes/home/data/sum.xlsx"
ORIGINAL_JSON_SPLIT_PATH = "/openbayes/home/data/inverse_split.json"
SUPPLEMENTARY_EXCEL_PATH = "/openbayes/home/data/base.xlsx"
SUPPLEMENTARY_GT_MAT_PATH = "/openbayes/home/data/base.mat"
OUTPUT_DIR = r"/openbayes/home/inverse/outputs_generated_supplementary"
VIS_DIR = r"/openbayes/home/inverse/visualizations_supplementary"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VIS_DIR, exist_ok=True)



def normalize_job_id(job_id_str):
    match = re.match(r'(Job-\d+)', job_id_str)
    if match:
        return match.group(1)
    return job_id_str

def get_fitted_scaler(original_excel_path, original_json_path):
    try:
        with open(original_json_path, 'r', encoding='utf-8') as f:
            data_split = json.load(f)
        train_job_ids = set(data_split.get('train', []))

        df_original = pd.read_excel(original_excel_path, header=0)
        original_columns = df_original.columns.astype(str)
        
    except Exception as e:
        print(f"❌ Error: Failed to load raw training data or JSON file. {e}")
        return None

    if not train_job_ids:
        print("❌ Error: No Job IDs found in the 'train' section of the JSON file.")
        return None

    train_vectors_smoothed = []
    for job_id in tqdm(train_job_ids, desc="Smoothen training vectors"):
        pattern = re.compile(f"^{re.escape(job_id)}([^0-9]|$)")
        found_col = next((col for col in original_columns if pattern.match(col)), None)
        
        if found_col is None:
            continue
        
        try:
            vector_raw = df_original[found_col].values[:301].astype(np.float32)
            if len(vector_raw) == 301:
                vector_smoothed = savgol_filter(vector_raw, 20, 2)
                train_vectors_smoothed.append(vector_smoothed)
        except Exception:
            continue
    
    if not train_vectors_smoothed:
        print("❌ Error: No valid training vectors extracted from the original Excel.")
        return None
        
    scaler = MinMaxScaler()
    scaler.fit(np.array(train_vectors_smoothed))
    print(f"✅ Scaler fitting completed! A total of {len(train_vectors_smoothed)} training samples were used.")
    return scaler

def load_and_process_supplementary_data(file_path, fitted_scaler):
    try:
        df = pd.read_excel(file_path, header=0)
    except FileNotFoundError:
        print(f"❌ Error: Excel file not found at {file_path}")
        return None, None

    job_columns = [col for col in df.columns if isinstance(col, str) and col.startswith('Job-')]
    if not job_columns:
        print(f"❌ Error: No valid 'Job-' columns found in file {file_path}.")
        return None, None
        
    all_vectors_raw = []
    all_job_ids = []
    for col in job_columns:
        job_id = normalize_job_id(col.strip())
        vector = df[col].dropna().values[:301].astype(np.float32)
        if len(vector) == 301:
            all_vectors_raw.append(vector)
            all_job_ids.append(job_id)

    if not all_vectors_raw:
        print("❌ Error: No valid 301-dimensional vectors loaded from supplementary Excel.")
        return None, None

    # 1. S-G smoothing
    print("Step a: Apply S-G smoothing to supplementary data...")
    all_vectors_smoothed = np.array([savgol_filter(vec, 20, 2) for vec in tqdm(all_vectors_raw, desc="Smoothen supplementary vectors")])

    # 2. Min-Max normalization (using the fitted Scaler)
    print("Step b: Apply fitted Scaler for normalization...")
    condition_vectors_processed = fitted_scaler.transform(all_vectors_smoothed)

    print(f"✅ Successfully loaded and processed {len(all_job_ids)} supplementary samples.")
    return condition_vectors_processed, all_job_ids


def generate_samples_from_vectors(ckpt_path, condition_vectors, top_p, temperature):
    """Load the specified model and generate samples from the directly passed, processed condition vectors."""
    print("\n--- Step 1: Start generating samples from processed supplementary data ---")
    print(f"Sampling strategy: Top-P = {top_p}, Temperature = {temperature}")
    config = OmegaConf.load(CONFIG_PATH)
    model = instantiate_from_config(config.model)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get('state_dict', checkpoint)
    model.load_state_dict(state_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    grid_h, grid_w = config.model.params.grid_size
    total_steps = grid_h * grid_w
    
    condition_tensor = torch.from_numpy(condition_vectors).float().to(device)

    with torch.no_grad():
        generated_indices = model.sample(
            condition=condition_tensor,
            steps=total_steps,
            batch_size=condition_tensor.shape[0],
            temperature=temperature,
            top_p=top_p
        )
        
    final_tensor = generated_indices
    output_numpy = final_tensor.squeeze(1).cpu().numpy() + 1
    output_numpy_final = output_numpy.transpose(1, 2, 0)
    
    output_filename_suffix = f"mode_topp_p{top_p}_t{temperature:.2f}"
    output_file_path = os.path.join(OUTPUT_DIR, f"generated_supplementary_{len(condition_vectors)}samples_{output_filename_suffix}.mat")
    sio.savemat(output_file_path, {"generated_indices": output_numpy_final})
    print(f"✅ Sample generation completed, file saved to: '{output_file_path}'")
    return output_file_path


def reconstruct_single_image(index_matrix, img_list, H, W, cell_H, cell_W):
    """Reconstruct a complete image from a single index matrix."""
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


def visualize_supplementary(generated_mat_path, gt_mat_path, job_ids, top_p, temperature):
    """
    Load ground truth and predicted results of supplementary data, and plot side-by-side comparison.
    """
    run_suffix = f"p{top_p}_t{temperature:.2f}"
    current_vis_dir = os.path.join(VIS_DIR, run_suffix)
    os.makedirs(current_vis_dir, exist_ok=True)
    print(f"Comparison images for this run will be saved to: '{current_vis_dir}'")

    try:
        pred_struct = sio.loadmat(generated_mat_path)
        pred_var_name = [k for k in pred_struct.keys() if not k.startswith('__')][0]
        predicted_data = pred_struct[pred_var_name]

        gt_struct = sio.loadmat(gt_mat_path)
        gt_var_name = [k for k in gt_struct.keys() if not k.startswith('__')][0]
        ground_truth_data = gt_struct[gt_var_name]

        img_list_struct = sio.loadmat(IMG_LIST_PATH)
        img_list_raw = img_list_struct['imgList']
        img_list = [cell for cell in img_list_raw[0]]

        print('✅ Comparison data loaded successfully.')
    except Exception as e:
        print(f"❌ Error loading comparison data: {e}")
        return

    pred_h, pred_w, pred_b = predicted_data.shape
    gt_b, gt_h, gt_w = ground_truth_data.shape

    print(f"🔍 Shape Diagnosis: Predicted Data (H, W, B) = ({pred_h}, {pred_w}, {pred_b})")
    print(f"🔍 Shape Diagnosis: Ground Truth Data (B, H, W) = ({gt_b}, {gt_h}, {gt_w})")

    if pred_h != gt_h or pred_w != gt_w:
        print(f"\n❌ Critical Error: Geometric dimensions of predicted results ({pred_h}, {pred_w}) do not match ground truth ({gt_h}, {gt_w})!")
        return
    if pred_b != gt_b:
        print(f"\n⚠️ Warning: Number of samples in predicted results ({pred_b}) does not match ground truth ({gt_b}). Will process according to the minimum sample size.")

    H, W = pred_h, pred_w
    num_samples = min(pred_b, gt_b, len(job_ids))
    cell_H, cell_W = img_list[0].shape

    for i in tqdm(range(num_samples), desc="Generating comparison images"):
        job_id = job_ids[i]
        
        predicted_grid = predicted_data[:, :, i]
        ground_truth_grid = ground_truth_data[i, :, :]
        
        gt_reconstructed = reconstruct_single_image(ground_truth_grid, img_list, H, W, cell_H, cell_W)
        pred_reconstructed = reconstruct_single_image(predicted_grid, img_list, H, W, cell_H, cell_W)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        axes[0].imshow(gt_reconstructed, cmap='gray')
        axes[0].set_title("Ground Truth")
        axes[0].axis('off')
        
        axes[1].imshow(pred_reconstructed, cmap='gray')
        axes[1].set_title("Predicted")
        axes[1].axis('off')
        
        fig.suptitle(f"Comparison for {job_id}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        output_filepath = os.path.join(current_vis_dir, f"{job_id}_comparison.png")
        plt.savefig(output_filepath, dpi=200)
        plt.close(fig)

def run_post_processing(ckpt_path, top_p, temperature):
    """Post-processing main process, integrating Scaler fitting and data preprocessing"""
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"❌ Error: Model checkpoint file does not exist at '{ckpt_path}'")
        return

    # 1: Fit Scaler
    scaler = get_fitted_scaler(ORIGINAL_EXCEL_PATH, ORIGINAL_JSON_SPLIT_PATH)
    if scaler is None:
        print("❌ Error: Failed to create Scaler, program terminated.")
        return

    # 2: Load and process supplementary data using the fitted Scaler
    condition_vectors, job_ids = load_and_process_supplementary_data(SUPPLEMENTARY_EXCEL_PATH, scaler)
    if condition_vectors is None:
        print("❌ Error: Failed to load or process data from Excel file, program terminated.")
        return

    # 3: Use processed data for model inference
    generated_mat_file = generate_samples_from_vectors(
        ckpt_path=ckpt_path,
        condition_vectors=condition_vectors,
        top_p=top_p,
        temperature=temperature
    )

    # 4: Visualize comparisons
    if generated_mat_file and os.path.exists(generated_mat_file):
        visualize_supplementary(
            generated_mat_path=generated_mat_file,
            gt_mat_path=SUPPLEMENTARY_GT_MAT_PATH,
            job_ids=job_ids,
            top_p=top_p,
            temperature=temperature
        )
    else:
        print("❌ Error: Failed to generate .mat file, skipping visualization step.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate predictions for supplementary data and compare with ground truth (integrating S-G smoothing and training set Scaler).")
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to the trained model checkpoint file (.ckpt)')
    parser.add_argument('--top_p', type=float, default=0.92, help='Top-P sampling probability threshold')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature coefficient')
    args = parser.parse_args()
    run_post_processing(args.ckpt_path, args.top_p, args.temperature)