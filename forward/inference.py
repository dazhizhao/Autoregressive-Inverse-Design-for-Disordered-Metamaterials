import torch
import pandas as pd
import numpy as np
import os
import glob
import scipy.io
from tqdm import tqdm
from FNO import FNO
from data_loader import FNODataModule


def find_checkpoint(directory: str) -> str:
    """Look for .ckpt files starting with 'best-' in the specified directory"""
    search_path = os.path.join(directory, "best-*.ckpt")
    checkpoints = glob.glob(search_path)
    if not checkpoints:
        raise FileNotFoundError(f"Error: Could not find any 'best-*.ckpt' weight file in the directory '{directory}'.")
    return max(checkpoints, key=os.path.getctime)


if __name__ == '__main__':
    temp = [0.25, 0.5, 0.75, 1, 1.25, 1.5]
    for i in temp:
        FNO_CHECKPOINT_DIR = r'/openbayes/home/forward/fno_32_6_8_checkpoints'\
        # path to the structures needed to be predicted
        NEW_MAT_FILE_PATH = f'/openbayes/home/inverse/outputs_generated/generated_clusters_200samples_mode_topp_p0.95_t{i:.2f}.mat' 
        OUTPUT_EXCEL_PATH = f'/openbayes/home/inverse/fno_inf/fno_t{i}.xlsx'
        ORIGINAL_EXCEL_FILE_PATH = r"/openbayes/home/data/sum.xlsx"
        ORIGINAL_JSON_FILE_PATH = r"/openbayes/home/data/new_forward_split.json"
        ORIGINAL_MAT_FILE_PATH = r"/openbayes/home/data/ind_mat_all.mat"

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {DEVICE}")

        try:
            fno_ckpt_path = find_checkpoint(FNO_CHECKPOINT_DIR)
            print(f"\nFound and loaded FNO weights: {fno_ckpt_path}")
            fno_model = FNO.load_from_checkpoint(fno_ckpt_path).to(DEVICE)
            fno_model.eval()
        except FileNotFoundError as e:
            print(e)
            exit()

        data_module = FNODataModule(
            excel_path=ORIGINAL_EXCEL_FILE_PATH,
            json_path=ORIGINAL_JSON_FILE_PATH,
            mat_path=ORIGINAL_MAT_FILE_PATH,
        )
        data_module.setup()
        scaler = data_module.scaler

        try:
            print(f"\nLoading the matrix to be predicted from '{NEW_MAT_FILE_PATH}'...")
            mat_data = scipy.io.loadmat(NEW_MAT_FILE_PATH)
            new_matrices = mat_data['generated_indices']
            num_samples = new_matrices.shape[2]
            print(f"Successful loading, found {num_samples} samples to be predicted.")
        except FileNotFoundError:
            print(f"Error: Could not find new .mat file '{NEW_MAT_FILE_PATH}'.")
            exit()
        except KeyError:
            print(f"Error: Could not find key 'ind_mat_all' in '{NEW_MAT_FILE_PATH}'. Please ensure the key name is correct.")
            exit()

        predictions_dict = {}
        with torch.no_grad():
            for i in tqdm(range(num_samples), desc="Predicting..."):
                matrix = new_matrices[:, :, i].astype(np.float32)
                input_tensor = torch.from_numpy(matrix).unsqueeze(0).unsqueeze(0).to(DEVICE)
                pred_normalized = fno_model(input_tensor)
                pred_original = scaler.inverse_transform(pred_normalized.cpu().numpy())
                job_id = f"Job-{i}"
                predictions_dict[job_id] = pred_original.flatten()

        print(f"\nPredictions are being saved to the: '{OUTPUT_EXCEL_PATH}'...")
        df_preds = pd.DataFrame(predictions_dict)
        df_preds.index.name = 'Dimension'
        df_preds.to_excel(OUTPUT_EXCEL_PATH, index=True)