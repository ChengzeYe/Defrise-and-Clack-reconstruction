import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import Projections_Dataset
from DandCReconstrucion import Pipeline


@hydra.main(version_base=None, config_path="config", config_name="config_test")
def main(cfg: DictConfig):
    params = dict(cfg)

    dataset = Projections_Dataset(**params)
    test_dataloader = DataLoader(dataset, batch_size=params['batch_size'], shuffle=False,
                                 num_workers=0)
    geometry = dataset.geom[0]
    print("Parameter loaded")

    model = Pipeline(geometry=geometry, learning_rate=params['base_lr'], num_data=dataset.__len__(),
                     num_epoch=params['num_epochs'])
    model = model.load_from_checkpoint(
        r"checkpoints/checkpoint.ckpt",
        geometry=geometry, learning_rate=params['base_lr'],
        num_data=dataset.__len__(), num_epoch=params['num_epochs'])

    callbacks = []

    wandb_logger = None

    print("Testing will be started now")
    trainer = pl.Trainer(max_epochs=params['num_epochs'], callbacks=callbacks, logger=wandb_logger, accelerator='gpu',
                         precision=32, fast_dev_run=False, log_every_n_steps=1, devices=1)

    output = trainer.predict(model, dataloaders=test_dataloader, return_predictions=True)
    output = output[0][0].numpy()
    ground_truth = dataset[0][1]
    show(preprocessing(output), preprocessing(ground_truth), geometry)


def preprocessing(recon_volume):
    """
        Normalize a numpy.ndarray using min-max scaling to bring all values into the range [0, 1].

        Parameters:
        - recon_volume (numpy.ndarray): The input array to normalize.

        Returns:
        - numpy.ndarray: The normalized numpy.ndarray with values scaled to [0, 1].
    """
    output = (recon_volume - np.min(recon_volume)) / (np.max(recon_volume) - np.min(recon_volume))
    return output


def show(output, ground_truth, geometry):
    """
        Display comparison images of reconstructed volume slices and ground truth slices.

        Parameters:
        - output (numpy.ndarray): The 3D array of the reconstructed volume.
        - ground_truth (numpy.ndarray): The 3D array of the ground truth volume.
        - geometry (object): An object containing the geometry configuration, specifically volume_shape.

        This function displays three orthogonal slices (axial, sagittal, and coronal) from both
        the output and ground truth for visual comparison.
    """
    plt.subplot(1, 2, 1)
    plt.imshow(output[geometry.volume_shape[1] // 2, :, :], cmap='gray')
    plt.title('reconstructed volume')
    plt.subplot(1, 2, 2)
    plt.imshow(ground_truth[geometry.volume_shape[1] // 2, :, :], cmap='gray')
    plt.title('Ground Truth')
    plt.legend()
    plt.show()
    plt.subplot(1, 2, 1)
    plt.imshow(output[:, geometry.volume_shape[1] // 2, :], cmap='gray')
    plt.title('reconstructed volume')
    plt.subplot(1, 2, 2)
    plt.imshow(ground_truth[:, geometry.volume_shape[1] // 2, :], cmap='gray')
    plt.title('Ground Truth')
    plt.legend()
    plt.show()
    plt.subplot(1, 2, 1)
    plt.imshow(output[:, :, geometry.volume_shape[1] // 2], cmap='gray')
    plt.title('reconstructed volume')
    plt.subplot(1, 2, 2)
    plt.imshow(ground_truth[:, :, geometry.volume_shape[1] // 2], cmap='gray')
    plt.title('Ground Truth')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    print("Test has started")
    main()
