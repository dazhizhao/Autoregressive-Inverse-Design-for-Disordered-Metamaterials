import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import json
import scipy.io
import re
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter

class FNODataset(Dataset):
    def __init__(self, excel_path: str, mat_path: str, job_ids: list):
        super().__init__()
        
        self.job_ids = job_ids
        
        mat_data = scipy.io.loadmat(mat_path)
        self.ground_truth_matrices = mat_data['ind_mat_all']

        self.df = pd.read_excel(excel_path, header=0) 
        
        self.data_map = []
        excel_columns = self.df.columns.astype(str)
        
        for job_id in self.job_ids:
            pattern = re.compile(f"^{re.escape(job_id)}([^0-9]|$)")
            found_col = next((col for col in excel_columns if pattern.match(col)), None)
            
            if found_col is None:
                print(f"Warning: The column corresponding to job {job_id} was not found in the Excel file and was skipped.")
                continue

            try:
                numerical_index = int(job_id.split('-')[1])
                
                condition_vector_raw = self.df[found_col].values[:301].astype(np.float32)
                
                if len(condition_vector_raw) != 301:
                    print(f"Warning: job {job_id} has less than 301 data points, actually {len(condition_vector_raw)}. Skipped.")
                    continue

                # Apply smoothing filter
                condition_vector_smoothed = savgol_filter(condition_vector_raw, 20, 2)

                matrix = self.ground_truth_matrices[:, :, numerical_index].astype(np.float32)

                self.data_map.append({
                    "job_id": job_id,
                    "matrix": matrix,
                    "condition_vector": condition_vector_smoothed.astype(np.float32)
                })
            except (ValueError, IndexError) as e:
                print(f"Warning: An error occurred while processing job {job_id}: {e}")
                continue

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):
        map_item = self.data_map[idx]
        input_matrix = torch.from_numpy(map_item['matrix']).unsqueeze(0)
        target_vector = torch.from_numpy(map_item['condition_vector'])
        return input_matrix, target_vector
    
class FNODataModule(pl.LightningDataModule):
    def __init__(self, excel_path, json_path, mat_path, batch_size=32, num_workers=4):
        super().__init__()
        self.excel_path = excel_path
        self.json_path = json_path
        self.mat_path = mat_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.scaler = MinMaxScaler()

    def setup(self, stage=None):
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data_split = json.load(f)
        
        train_job_ids = data_split.get('train', [])
        val_job_ids = data_split.get('val', [])

        if not train_job_ids or not val_job_ids:
            raise ValueError(f"Error: JSON file '{self.json_path}' must contain 'train' and 'val' lists.")

        self.train_dataset = FNODataset(
            excel_path=self.excel_path,
            mat_path=self.mat_path,
            job_ids=train_job_ids
        )
        
        self.val_dataset = FNODataset(
            excel_path=self.excel_path,
            mat_path=self.mat_path,
            job_ids=val_job_ids
        )
        
        # Fitting a Scaler with an Explicitly Defined Training Set
        if len(self.train_dataset) > 0:
            train_targets = np.array([item['condition_vector'] for item in self.train_dataset.data_map])
            self.scaler.fit(train_targets)
            print("Scaler fitting completed!")
        else:
            print("Warning: Training set is empty, unable to fit Scaler.")

        # Store the job_ids of the validation set for later use (e.g., visualization)
        self.val_job_ids = [item['job_id'] for item in self.val_dataset.data_map]

    def custom_collate_fn(self, batch):
        inputs, targets_raw = zip(*batch)
        
        inputs_tensor = torch.stack(inputs, 0)
        
        targets_raw_np = torch.stack(targets_raw, 0).numpy()
        targets_normalized_np = self.scaler.transform(targets_raw_np)
        targets_tensor = torch.from_numpy(targets_normalized_np).float()
        
        return inputs_tensor, targets_tensor

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.custom_collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.custom_collate_fn
        )