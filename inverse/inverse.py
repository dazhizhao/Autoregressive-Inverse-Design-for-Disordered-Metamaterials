import os
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from main import instantiate_from_config
from data_loader import InverseDataModule

# AR model config path
CONFIG_PATH = r"/openbayes/home/configs/config.yaml"
TENSORBOARD_LOG_DIR = "/openbayes/home/tf_dir" 
OUTPUT_DIR = "/openbayes/home/inverse"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")

POST_PROCESS_TOP_P = 0.92
POST_PROCESS_TEMP = 1.0

os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def train():
    # --- 1. Load configuration ---
    config = OmegaConf.load(CONFIG_PATH)

    # --- 2. Instantiation of data modules and model---
    data_params = config.data.params
    data_module = InverseDataModule(
        excel_path=data_params.excel_path,
        json_path=data_params.json_path,
        mat_path=data_params.mat_path,
        batch_size=data_params.batch_size,
        num_workers=data_params.get("num_workers", 4)
    )
    model = instantiate_from_config(config.model)

    # --- 4. Configure callbacks ---
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        monitor="val_loss",
        filename='best-ar-model-{epoch:02d}-{val_loss:.4f}',
        mode="min",
        save_top_k=1,
        save_last=True,
        verbose=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    logger = TensorBoardLogger(save_dir=TENSORBOARD_LOG_DIR, name="inverse_model_logs")

    use_gpu = torch.cuda.is_available()
    accelerator = "gpu" if use_gpu else "cpu"
    devices = 1 if use_gpu else None

    trainer_params = config.get("lightning", {}).get("trainer", {})
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=trainer_params.get("max_epochs", 50),
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
    )

    model.learning_rate = config.model.base_learning_rate
    
    # --- 5. Model training ---
    trainer.fit(model=model, datamodule=data_module)
    print(f"Best model path: {checkpoint_callback.best_model_path}")
    

if __name__ == '__main__':
    pl.seed_everything(126, workers=True)
    train()