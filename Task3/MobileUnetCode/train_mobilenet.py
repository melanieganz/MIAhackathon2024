import torch
from torch import nn
import torch.quantization
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from mobileunet_model import MobileUnet
from dataloaders import get_dataloaders
from utils import read_yaml_file
# from carbontracker.tracker import CarbonTracker


def dice_score(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    y_pred = y_pred.round().clip(0,1)
    y_pred = y_pred.long()
    y_true = y_true.long()
    score = 2*(y_pred*y_true).sum() / (y_pred.sum() + y_true.sum())
    return score.item()

class FastSegModel(pl.LightningModule):
    def __init__(self, config):
        super(FastSegModel, self).__init__()
        # Define your model
        self.config = config
        self.model = MobileUnet(config["num_classes"])
        self.criterion = nn.BCEWithLogitsLoss()
        # self.accuracy = Accuracy()
        # self.dice = Dice(num_classes=config["num_classes"])
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        self.log('train_loss', loss, on_epoch=True, on_step=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['label']
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        dice = dice_score(torch.sigmoid(logits), y)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=True)
        self.log('val_dice', dice, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def configure_optimizers(self):
        # Create a quantized optimizer
        optimizer = Adam(self.parameters(), lr=0.001)
        
        # Use a scheduler that doesn't require setting the learning rate explicitly
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'frequency': 1
            }
        }


def main(config):
    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=True, mode='min')
    logger = TensorBoardLogger("logs")
    # Create a checkpoint callback

    checkpoint_callback = ModelCheckpoint(
        dirpath=config['model_save_dir'],
        filename=f"{config['modality_type']}",
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )

    # Initialize the model
    model = FastSegModel(config)
    train_loader, val_loader = get_dataloaders(config)
    # Initialize the trainer with mixed precision (AMP) and early stopping
    trainer = pl.Trainer(
        logger = logger,
        max_epochs=config['num_epochs'],  # you can set this higher, early stopping will halt training
        callbacks=[checkpoint_callback, early_stopping],
        precision='16-mixed',  # Use AMP
        # accelerator='gpu',  # Use GPU if available
        # devices=1  # Number of GPUs
    )

    # Train the model
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == '__main__':

    for modality in ['FMRI', 'T2W', 'DWI']:
        config = read_yaml_file("config.yaml")
        # iterate over different modalities and train the model
        config["modality_type"] = modality
        try:
            main(config)
        except:
            print(f"data for {modality} modality is not available")
