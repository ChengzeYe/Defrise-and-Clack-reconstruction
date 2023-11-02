import os
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor,EarlyStopping
from torch.utils.data import DataLoader
from dataset import Projections_Dataset
import matplotlib.pyplot as plt
import numpy as np
from DandCReconstrucion import Pipeline
import torch
'''import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:6144"'''

# wandb agent lschneid/Multi_Order_SpearmanLoss/9ags44yq --count=5



@hydra.main(version_base=None, config_path="config", config_name="config_test")
def main(cfg : DictConfig):
    params = dict(cfg)
    dataset = Projections_Dataset(**params)
    train_dataset, validation_dataset = train_test_split(dataset, test_size=params['test_size'])
    train_dataloader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=0)
    validation_dataloader = DataLoader(validation_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=0)
    geometry = dataset.geom[0]
    print("Parameter loaded")
    model = Pipeline(geometry=geometry,learning_rate=params['base_lr'], num_data = dataset.__len__(), num_epoch = params['num_epochs'])

    model_checkpoint = ModelCheckpoint(dirpath=params['checkpoints_path'],
                                       filename="checkpoint",
                                       verbose=True, monitor='Validation Loss', mode='min')
    # lr_monitor = LearningRateMonitor()
    early_stop_callback = EarlyStopping(monitor="Validation Loss", min_delta=0.00, patience=3, mode="min")

    callbacks = [model_checkpoint, early_stop_callback]#, lr_monitor]#,early_stop_callback] # more callbacks can be added
    wandb_logger = None
    # TODO: add wandb log Routine
    print("Training will be started now")
    trainer = pl.Trainer(max_epochs=params['num_epochs'], callbacks=callbacks,logger=wandb_logger,  accelerator='gpu', precision=32, #move_metrics_to_cpu = True, , strategy="ddp"
                        fast_dev_run=False, log_every_n_steps=1, devices=1)#,check_val_every_n_epoch=1,check_val_every_n_epoch=5), ,overfit_batches=4val_check_interval=0.2  #reload_dataloaders_every_n_epochs = 20, ,overfit_batches=4)#

    #trainer.fit(model,train_dataloaders=train_dataloader, val_dataloaders=validation_dataloader, ckpt_path=None)
    trainer.predict(model,dataloaders=validation_dataloader)
    train_loss = model.train_loss
    validation_loss= model.validation_loss
    plt.plot(np.arange(len(train_loss)), train_loss, label='train loss')
    plt.plot(np.arange(len(validation_loss)), validation_loss, label='val loss')
    plt.yscale('log')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    print("Trainer has started")
    main()


