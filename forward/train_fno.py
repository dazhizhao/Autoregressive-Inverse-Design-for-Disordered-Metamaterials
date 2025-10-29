import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import matplotlib.pyplot as plt
import os
import numpy as np
from FNO import FNO
from data_loader import FNODataModule


def visualize_predictions(model, dataloader, scaler, device, output_dir, val_job_ids):
    """
    Make predictions on the validation set and visualize the results.
    """
    print(f"\n--- Start generating validation set prediction plots, saving to '{output_dir}' ---")
    os.makedirs(output_dir, exist_ok=True)
    
    model.to(device)
    model.eval()
    
    sample_idx = 0
    with torch.no_grad():
        for batch in dataloader:
            inputs, targets_normalized = batch
            inputs = inputs.to(device)
            preds_normalized = model(inputs).cpu().numpy()
            preds_original = scaler.inverse_transform(preds_normalized)
            targets_original = scaler.inverse_transform(targets_normalized.numpy())
            
            for i in range(preds_original.shape[0]):
                job_id = val_job_ids[sample_idx]
                pred_vec = preds_original[i]
                true_vec = targets_original[i]
                
                plt.figure(figsize=(12, 6))
                plt.plot(true_vec, label='Ground Truth', color='blue', alpha=0.7)
                plt.plot(pred_vec, label='Prediction', color='red', linestyle='--')
                plt.title(f'{job_id}')
                plt.xlabel('Displacement')
                plt.ylabel('Force')
                plt.legend()
                plt.grid(True, linestyle=':', alpha=0.6)
                
                save_path = os.path.join(output_dir, f'{job_id}.png')
                plt.savefig(save_path)
                plt.close()
                
                sample_idx += 1


if __name__ == '__main__':
    pl.seed_everything(126, workers=True)
    
    EXCEL_FILE_PATH = "/openbayes/home/data/sum.xlsx"
    JSON_FILE_PATH = "/openbayes/home/data/new_forward_split.json"
    MAT_FILE_PATH = "/openbayes/home/data/ind_mat_all.mat"
    log_path = '/openbayes/home/tf_dir'
    
    BATCH_SIZE = 1
    MAX_EPOCHS = 100
    LEARNING_RATE = 1e-3
    
    FNO_LATENT_CHANNELS = 32
    FNO_MODES = 6
    FNO_LAYERS = 8 

    data_module = FNODataModule(
        excel_path=EXCEL_FILE_PATH,
        json_path=JSON_FILE_PATH,
        mat_path=MAT_FILE_PATH,
        batch_size=BATCH_SIZE
    )

    model = FNO(
        in_channels=1,
        input_height=10,
        input_width=10,
        fno_latent_channels=FNO_LATENT_CHANNELS,
        fno_layers=FNO_LAYERS,
        fno_modes=FNO_MODES,
        learning_rate=LEARNING_RATE,
        output_dim=301 
    )

    logger = TensorBoardLogger(log_path, name=f"fno_with_new_json")
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath=f'fno_{FNO_LATENT_CHANNELS}_{FNO_MODES}_{FNO_LAYERS}_checkpoints',
        filename='best-fno-model-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        mode='min',
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=20,
        verbose=True,
        mode='min'
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator='auto',
        devices='auto',
        logger=logger,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        deterministic=True
    )
    
    trainer.fit(model, data_module)

    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        best_model = FNO.load_from_checkpoint(best_model_path)
        
        data_module.setup('validate')
        val_loader = data_module.val_dataloader()
        scaler = data_module.scaler
        val_job_ids = data_module.val_job_ids
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        visualize_predictions(
            model=best_model,
            dataloader=val_loader,
            scaler=scaler,
            device=device,
            output_dir=f'./fno_{FNO_LATENT_CHANNELS}_{FNO_MODES}_{FNO_LAYERS}_predictions',
            val_job_ids=val_job_ids
        )
    else:
        print("\n fails to find the optimal model, skip the visualization.")