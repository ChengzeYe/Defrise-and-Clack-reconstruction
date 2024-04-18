import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from dataset import Projections_Dataset
from DandCReconstrucion import Pipeline


@hydra.main(version_base=None, config_path="config", config_name="config_test")
def main(cfg: DictConfig):
    params = dict(cfg)

    dataset = Projections_Dataset(**params)
    train_dataset, validation_dataset = train_test_split(dataset, test_size=params['test_size'])
    train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=0)
    validation_dataloader = DataLoader(validation_dataset, batch_size=params['batch_size'], shuffle=False,
                                       num_workers=0)
    geometry = dataset.geom[0]
    print("Parameter loaded")

    model = Pipeline(geometry=geometry, learning_rate=params['base_lr'], num_data=train_dataset.__len__(),
                     num_epoch=params['num_epochs'])

    model_checkpoint = ModelCheckpoint(dirpath=params['checkpoints_path'],
                                       filename="checkpoint",
                                       verbose=True, monitor='Validation Loss', mode='min')

    early_stop_callback = EarlyStopping(monitor="Validation Loss", min_delta=0.00, patience=3, mode="min")

    callbacks = [model_checkpoint, early_stop_callback]

    wandb_logger = None

    print("Training will be started now")
    trainer = pl.Trainer(max_epochs=params['num_epochs'], callbacks=callbacks, logger=wandb_logger, accelerator='gpu',
                         precision=32, fast_dev_run=False, log_every_n_steps=1, devices=1)
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader)


if __name__ == "__main__":
    print("Trainer has started")
    main()
