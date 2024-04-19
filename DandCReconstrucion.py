import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.functional as TF
import torchmetrics
import pytorch_lightning as pl
from pyronn.ct_reconstruction.layers.backprojection_2d import ParallelBackProjection2D
from pyronn.ct_reconstruction.layers.backprojection_3d import ConeBackProjection3D
from weight import weights_3d
from intermediateFunction import geometry_radon_2d


class Pipeline(pl.LightningModule):
    def __init__(self, geometry, learning_rate, num_data, num_epoch):
        super().__init__()
        self.num_data = num_data
        self.geometry = geometry
        self.num_epoch = num_epoch
        self.geom_2d = geometry_radon_2d(geometry)
        self.learningRate = learning_rate

        # self.weight_init = -torch.rand((self.geom_2d.number_of_projections, self.geom_2d.detector_shape[-1]))  # When the orbit is a circular orbit
        # self.weight_init = -torch.rand((self.geometry.number_of_projections, self.geom_2d.number_of_projections, self.geom_2d.detector_shape[-1]))  # When the orbit is a non-circular orbit
        self.weight_init = torch.tensor(weight_initialization(self.geom_2d, D=self.geometry.source_isocenter_distance))  # Use analytic redundancy weight

        self.DandCrecon = DandCrecon(geometry=self.geometry, weight_init=self.weight_init, geom_2d=self.geom_2d)
        self.loss_fn = torch.nn.MSELoss()
        self._train_loss_agg = torchmetrics.MeanMetric()
        self._validation_loss_agg = torchmetrics.MeanMetric()
        self.train_loss = []
        self.validation_loss = []
        self.output = None
        self.ground_truth = None

    def forward(self, sinogram):
        recon_image = self.DandCrecon(sinogram)
        return recon_image

    def training_step(self, batch, batch_idx):
        ground_truth = preprocessing(batch[1])
        output = self.forward(batch[0])
        output = preprocessing(output)
        loss = self.loss_fn(output, ground_truth)
        self._train_loss_agg.update(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self.ground_truth = preprocessing(batch[1])
        self.output = self.forward(batch[0])
        self.output = preprocessing(self.output)

        reco = self.output
        show(reco[0, int(self.geometry.volume_shape[0] / 2), :, :], 'yz')
        show(reco[0, :, int(self.geometry.volume_shape[1] / 2), :], 'xz')
        show(reco[0, :, :, int(self.geometry.volume_shape[2] / 2)], 'xy')
        reco1 = self.ground_truth
        show(reco1[0, int(self.geometry.volume_shape[0] / 2), :, :], 'yz')
        show(reco1[0, :, int(self.geometry.volume_shape[1] / 2), :], 'xz')
        show(reco1[0, :, :, int(self.geometry.volume_shape[2] / 2)], 'xy')

        loss = self.loss_fn(self.output, self.ground_truth)
        self._validation_loss_agg.update(loss)
        return loss

    def on_train_epoch_end(self):
        train_loss_value = self._train_loss_agg.compute().cpu().detach().numpy()
        self.log("Train Loss", float(train_loss_value))
        self.train_loss.append(train_loss_value)
        print("Train loss", train_loss_value)
        self._train_loss_agg.reset()

    def on_validation_epoch_end(self):
        validation_loss_value = self._validation_loss_agg.compute().cpu().detach().numpy()
        self.log("Validation Loss", float(validation_loss_value))
        self.validation_loss.append(validation_loss_value)
        print("Validation loss", validation_loss_value)
        self._validation_loss_agg.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.learningRate * 10,
                                                        steps_per_epoch=self.num_data, epochs=self.num_epoch)
        return [optimizer], [scheduler]


class DandCrecon(torch.nn.Module):
    def __init__(self, geometry, weight_init, geom_2d):
        super(DandCrecon, self).__init__()
        self.geometry = geometry
        self.weight = torch.nn.Parameter(weight_init.float(), requires_grad=True)
        self.geom_2d = geom_2d
        self.backprojection_2d = ParallelBackProjection2D()
        self.backprojection_3d = ConeBackProjection3D()
        self.weight_3d = torch.tensor(weights_3d(self.geom_2d, D=self.geometry.source_detector_distance).copy()).cuda()
        self.relu = torch.nn.ReLU()

    def forward(self, sinogram):
        if isinstance(sinogram, list):
            sinogram = sinogram[0]
            weight = TF.gaussian_blur(self.weight.unsqueeze(0).unsqueeze(0), kernel_size=21, sigma=5).squeeze(
                0).squeeze(0)

        else:
            sinogram = sinogram
            weight = self.weight

        sinogram = torch.multiply(sinogram, weight)

        sinogram = torch.squeeze(sinogram, dim=0)
        sinogram = torch.gradient(sinogram, dim=2)[0]

        projection = self.backprojection_2d(sinogram.contiguous(), **self.geom_2d)

        projection = torch.multiply(projection, self.weight_3d)
        projection = torch.unsqueeze(projection, dim=0)

        reco = self.backprojection_3d(projection.contiguous(), **self.geometry)
        return self.relu(reco)


def weight_initialization(geom_2dm, D):
    """
        Initialize redundancy weight for circular orbit geometry.

        Parameters:
        - geom_2d (Geometry): The Geometry object for 2D Radon Transform.
        - D (float): Distance from the source to the isocenter.

        Returns:
        - numpy.ndarray: An array of redundancy weight for circular orbit geometry.
    """
    c = -1 / (8 * np.pi ** 2)
    s = geom_2dm.detector_shape[-1]
    cs = -(s - 1) / 2 * geom_2dm.detector_spacing[-1]
    angular_increment = 2 * np.pi / geom_2dm.number_of_projections
    sd2 = D ** 2
    w = np.zeros((geom_2dm.number_of_projections, geom_2dm.detector_shape[-1]), dtype=np.float32)
    for mu in range(0, geom_2dm.number_of_projections):
        cosmu = np.abs(np.cos(mu * angular_increment - np.pi / 2))
        for s in range(0, geom_2dm.detector_shape[-1]):
            ds = (s * geom_2dm.detector_spacing[-1] + cs) ** 2
            w[mu, s] = cosmu * sd2 / (sd2 + ds)
    return w * c


def preprocessing(recon_volume):
    """
        Normalize a tensor using min-max scaling to bring all values into the range [0, 1].

        Parameters:
        - recon_volume (torch.Tensor): The input tensor to normalize.

        Returns:
        - torch.Tensor: The normalized tensor with values scaled to [0, 1].
        """
    output = (recon_volume - torch.min(recon_volume)) / (torch.max(recon_volume) - torch.min(recon_volume))
    return output


def show(a, name):
    """
        Display a preprocessed image tensor using Matplotlib.

        Parameters:
        - a (torch.Tensor): The image tensor to be displayed.
        - name (str): The title of the figure to display.
        """
    a = preprocessing(a)
    plt.figure()
    plt.imshow(a.cpu().detach().numpy(), cmap='gray')
    plt.show()
    plt.axis('on')
    plt.close()


if __name__ == '__main__':
    pass
