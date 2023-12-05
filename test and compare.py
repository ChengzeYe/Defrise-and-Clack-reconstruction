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
from visualization import visualization1
'''import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:6144"'''

# wandb agent lschneid/Multi_Order_SpearmanLoss/9ags44yq --count=5


@hydra.main(version_base=None, config_path="config", config_name="config_test")
def main(cfg : DictConfig):
    params = dict(cfg)
    dataset = Projections_Dataset(**params)
    test_dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=False, num_workers=0)
    ground_truth = dataset[0][1]
    geometry = dataset.geom[0]
    print("Parameter loaded")
    model = Pipeline(geometry=geometry,learning_rate=params['base_lr'], num_data=dataset.__len__(), num_epoch=params['num_epochs'])
    #model = model.load_from_checkpoint("E:\MasterArbeit\code\Defrise-and-Clack-reconstruction\checkpoints\checkpoint-v1.ckpt", geometry=geometry, learning_rate=params['base_lr'], num_data=dataset.__len__(), num_epoch=params['num_epochs'])

    model_checkpoint = ModelCheckpoint(dirpath=params['checkpoints_path'],
                                       filename="checkpoint",
                                       verbose=True, monitor='Validation Loss', mode='min')
    callbacks = [model_checkpoint]
    wandb_logger = None
    print("Testing will be started now")
    trainer = pl.Trainer(max_epochs=params['num_epochs'], callbacks=callbacks,logger=wandb_logger,  accelerator='gpu', precision=32, #move_metrics_to_cpu = True, , strategy="ddp"
                        fast_dev_run=False, log_every_n_steps=1, devices=1)#,check_val_every_n_epoch=1,check_val_every_n_epoch=5), ,overfit_batches=4val_check_interval=0.2  #reload_dataloaders_every_n_epochs = 20, ,overfit_batches=4)#
    output = trainer.predict(model, dataloaders=test_dataloader, return_predictions=True)#torch.unsqueeze(dataset[0][0], dim=0)
    learned_weight = model.DandCrecon.weight.detach().cpu().numpy()
    output = preprocessing(output[0].cpu().detach().numpy())
    reco = output
    show(reco[0, 200, :, :], 'yz')
    show(reco[0, :, int(geometry.volume_shape[1] / 2), :], 'xz')
    show(reco[0, :, :, int(geometry.volume_shape[2] / 2)], 'xy')
    reco1 = ground_truth
    show(reco1[200, :, :], 'yz')
    show(reco1[:, int(geometry.volume_shape[1] / 2), :], 'xz')
    show(reco1[:, :, int(geometry.volume_shape[2] / 2)], 'xy')
    visualization1(learned_weight)
    #print("PReLU parameter: ", model.prelu.weight.detach().cpu().numpy())


def preprocessing(recon_volume):
    output = (recon_volume - np.min(recon_volume)) / (np.max(recon_volume) - np.min(recon_volume))
    return output

def show(a, name):
    a = preprocessing(a)
    plt.figure()
    plt.imshow(a, cmap='gray')  # plt.get_cmap('gist_gray')
    plt.show()
    plt.axis('on')
    plt.close()


if __name__ == "__main__":
    print("Trainer has started")
    main()