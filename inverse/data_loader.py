import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pandas as pd
import numpy as np
import json
import scipy.io
import re
import pytorch_lightning as pl
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import savgol_filter

class InverseDataset(Dataset):
    """
    Prepare data for the reverse design task.
    - Read 301-dimensional conditional vectors and perform S-G smoothing.
    - Provide (smoothed conditional vector, target grid).
    - Note: Normalization will be handled in collate_fn of DataModule.
    """
    def __init__(self, excel_path, json_path, mat_path):
        super().__init__()

        # 1. Load JSON file and get all Job IDs
        with open(json_path, 'r', encoding='utf-8') as f:
            data_split = json.load(f)

        # To accurately partition in DataModule later, load all IDs here
        all_job_ids = data_split.get('train', [])
        val_data = data_split.get('val', {})
        for cluster_name in sorted(val_data.keys()):
            all_job_ids.extend(val_data[cluster_name])
            
        self.job_ids = all_job_ids

        # 2. Loading .mat and .xlsx files
        mat_data = scipy.io.loadmat(mat_path)
        self.ground_truth_data = mat_data['ind_mat_all']
        self.df = pd.read_excel(excel_path, header=0)

        # 3. Preprocessing and mapping
        self.data_map = []
        excel_columns = self.df.columns.astype(str)

        for job_id in self.job_ids:
            pattern = re.compile(f"^{re.escape(job_id)}([^0-9]|$)")
            found_col = next((col for col in excel_columns if pattern.match(col)), None)
            
            if found_col is None:
                continue

            try:
                condition_vector_raw = self.df[found_col].values[:301].astype(np.float32)
                
                if len(condition_vector_raw) != 301:
                    print(f"Warning: The condition vector length of job {job_id} is not 301, skip the job.")
                    continue
                
                # Apply S-G smoothing filter
                condition_vector_smoothed = savgol_filter(condition_vector_raw, 20, 2)

                numerical_index = int(job_id.split('-')[1])
                ground_truth_grid = self.ground_truth_data[:, :, numerical_index] - 1

                self.data_map.append({
                    "job_id": job_id,
                    "condition_vector": condition_vector_smoothed.astype(np.float32),
                    "ground_truth_grid": ground_truth_grid.astype(np.int64) 
                })
            except (ValueError, IndexError) as e:
                print(f"Error occurred while processing job {job_id}: {e}")
                continue

    def __len__(self):
        return len(self.data_map)

    def __getitem__(self, idx):
        map_item = self.data_map[idx]

        # Return the smoothed but unnormalized vector and the target grid
        condition_vector = torch.from_numpy(map_item['condition_vector'])
        ground_truth_grid = torch.from_numpy(map_item['ground_truth_grid'])
        
        return condition_vector, ground_truth_grid

class InverseDataModule(pl.LightningDataModule):
    def __init__(self, excel_path, json_path, mat_path, batch_size=32, num_workers=4):
        super().__init__()
        self.excel_path = excel_path
        self.json_path = json_path
        self.mat_path = mat_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.scaler = MinMaxScaler()

    def setup(self, stage=None):
        # 1. Instantiate the full dataset
        full_dataset = InverseDataset(
            excel_path=self.excel_path,
            json_path=self.json_path,
            mat_path=self.mat_path
        )

        # 2. Strictly re-divide the training and validation sets according to the JSON file
        with open(self.json_path, 'r', encoding='utf-8') as f:
            data_split = json.load(f)
        
        train_job_ids = set(data_split['train'])
        val_job_ids = set()
        val_data = data_split.get('val', {})
        for cluster_name in sorted(val_data.keys()):
            val_job_ids.update(val_data[cluster_name])

        # Find the index of each ID in full_dataset
        train_indices = [i for i, item in enumerate(full_dataset.data_map) if item['job_id'] in train_job_ids]
        val_indices = [i for i, item in enumerate(full_dataset.data_map) if item['job_id'] in val_job_ids]

        # 3. Fit the Scaler on the training set
        train_vectors = np.array([full_dataset.data_map[i]['condition_vector'] for i in train_indices])
        self.scaler.fit(train_vectors)

        # 4. Create Subset
        self.train_dataset = Subset(full_dataset, train_indices)
        self.val_dataset = Subset(full_dataset, val_indices)

    def custom_collate_fn(self, batch):
        vectors, grids = zip(*batch)

        # 1. Normalize condition vectors
        vectors_np = torch.stack(vectors, 0).numpy()
        vectors_normalized_np = self.scaler.transform(vectors_np)
        vectors_tensor = torch.from_numpy(vectors_normalized_np).float()

        # 2. Stack target grids
        grids_tensor = torch.stack(grids, 0)

        return vectors_tensor, grids_tensor

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