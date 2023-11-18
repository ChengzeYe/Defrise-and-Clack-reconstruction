# import numpy as np
import numpy as np
import torch
import torchmetrics
import pytorch_lightning as pl
from pyronn.ct_reconstruction.layers.backprojection_2d import ParallelBackProjection2D
from pyronn.ct_reconstruction.layers.backprojection_3d import ConeBackProjection3D
from weight import weights_3d
import matplotlib.pyplot as plt
from intermediateFunction import intermediate_function, geometry_radon_2d
from visualization import visualization1


class Pipeline(pl.LightningModule):
    def __init__(self, geometry, learning_rate, num_data, num_epoch):
        super().__init__()
        self.num_data = num_data
        self.geometry = geometry
        self.num_epoch = num_epoch
        self.geom_2d = geometry_radon_2d(geometry)
        self.learningRate = learning_rate
        #self.weight_init = -torch.ones((geometry.number_of_projections, self.geom_2d.number_of_projections, self.geom_2d.detector_shape[-1]))  # 这里应该是个3D的
        self.weight_init = -torch.rand((self.geom_2d.number_of_projections, self.geom_2d.detector_shape[-1]))
        #self.weight_init = torch.tensor(weight_initialization(self.geom_2d, D=self.geometry.source_detector_distance))
        #self.weight_init = torch.tensor(np.load(r'E:\MasterArbeit\code\Defrise-and-Clack-reconstruction\redundancy_weight_60_100_circular.npy'))
        #self.weight_init1 = torch.tensor(np.load(r'E:\MasterArbeit\code\Defrise-and-Clack-reconstruction\redundancy_weight.npy'))

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
        sinogram = batch[0]
        # ground_truth = batch[1]
        ground_truth = preprocessing(batch[1])
        output = self.forward(sinogram)
        output = preprocessing(output)
        loss = self.loss_fn(output, ground_truth)
        self._train_loss_agg.update(loss)
        return loss

    def validation_step(self, batch, batch_idx):
        sinogram = batch[0]
        self.ground_truth = preprocessing(batch[1])
        self.output = self.forward(sinogram)
        self.output = preprocessing(self.output)

        '''reco = self.output
        show(reco[0, int(self.geometry.volume_shape[0] / 2), :, :], 'yz')
        show(reco[0, :, int(self.geometry.volume_shape[1] / 2), :], 'xz')
        show(reco[0, :, :, int(self.geometry.volume_shape[2] / 2)], 'xy')
        reco1 = self.ground_truth
        show(reco1[0, int(self.geometry.volume_shape[0] / 2), :, :], 'yz')
        show(reco1[0, :, int(self.geometry.volume_shape[1] / 2), :], 'xz')
        show(reco1[0, :, :, int(self.geometry.volume_shape[2] / 2)], 'xy')'''

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
        # log images to wandb
        if self.current_epoch % 5 == 0:
            print("PReLU parameter: ", self.DandCrecon.prelu.weight.detach().cpu().numpy())

            visualization1(self.DandCrecon.weight.cpu().detach().numpy())

            plt.subplot(1, 2, 1)
            plt.imshow(self.output[0, self.geometry.volume_shape[-1] // 2, :, :].cpu().detach().numpy(), cmap='gray')
            plt.title('reconstructed volume')
            plt.subplot(1, 2, 2)
            plt.imshow(self.ground_truth[0, self.geometry.volume_shape[-1] // 2, :, :].cpu().detach().numpy(), cmap='gray')  # 替换 'additional_data' 变量
            plt.title('Ground Truth')
            plt.legend()
            plt.savefig('reconstructed volume.png')
            #plt.show()
            plt.clf()
            self.logger.log_image(key="3D Surface Plot(learned weight)", images=['3D Surface Plot(learned weight).png'],
                                  caption=[f"3D_Surface_Plot(learned_weight)_{self.current_epoch}"])
            self.logger.log_image(key="learned weight line profile", images=['learned weight line profile.png'],
                                  caption=[f"learned_weight_line_profile_{self.current_epoch}"])
            self.logger.log_image(key="reconstructed volume", images=['reconstructed volume.png'],
                                  caption=[f"reconstructed_volume_{self.current_epoch}"])



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learningRate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=self.num_data, epochs=self.num_epoch)
        return [optimizer], [scheduler]


class DandCrecon(torch.nn.Module):
    def __init__(self, geometry, weight_init, geom_2d):
        super(DandCrecon, self).__init__()
        self.geometry = geometry
        self.weight = torch.nn.Parameter(weight_init.float(), requires_grad=True)
        self.geom_2d = geom_2d
        self.backprojection_2d = ParallelBackProjection2D()
        self.backprojection_3d = ConeBackProjection3D()
        self.prelu = torch.nn.PReLU(num_parameters=1, init=0.25)

    def forward(self, sinogram):#这里predict 和 fit 输入进来的不同 predict的时候需要[0]
        '''plt.imshow(sinogram[0][0].cpu())
        plt.show()'''
        weighted_sinogram = torch.multiply(sinogram[0], self.weight)
        derivative = torch.gradient(weighted_sinogram, dim=3)[0]
        derivative = torch.squeeze(derivative, dim=0)
        '''plt.imshow(derivative[0].cpu())
        plt.show()'''
        CB_projection = self.backprojection_2d(derivative.contiguous(), **self.geom_2d)
        weight_3d = torch.tensor(weights_3d(self.geom_2d, D=self.geometry.source_detector_distance).copy()).cuda()
        weighted_CB_projection = torch.multiply(CB_projection, weight_3d)
        '''plt.imshow(weighted_CB_projection[0].cpu())
        plt.show()'''
        weighted_CB_projection = torch.unsqueeze(weighted_CB_projection, dim=0)
        reco = self.backprojection_3d(weighted_CB_projection.contiguous(), **self.geometry)
        reco = self.prelu(reco)
#.cpu().numpy()[0]

        return reco


def weight_initialization(geom_2dm, D):
    c = -1/(8*np.pi**2)
    s = geom_2dm.detector_shape[-1]
    cs = -(s - 1) / 2 * geom_2dm.detector_spacing[-1]
    angular_increment = 2 * np.pi / geom_2dm.number_of_projections
    # mus = -(geom_2dm.number_of_projections - 1) / 2 * angular_increment
    sd2 = D ** 2
    w = np.zeros((geom_2dm.number_of_projections, s), dtype=np.float32)
    for mu in range(0, geom_2dm.number_of_projections):
        a = np.abs(np.cos(mu*angular_increment-np.pi/2))#
        for s in range(0, geom_2dm.detector_shape[-1]):
            ds = (s * geom_2dm.detector_spacing[-1] + cs) ** 2
            w[mu, s] = a*sd2/(sd2+ds)
    return w*c


def preprocessing(recon_volume):
    output = (recon_volume - torch.min(recon_volume)) / (torch.max(recon_volume) - torch.min(recon_volume))
    return output


def show(a, name):
    a = preprocessing(a)
    plt.figure()
    plt.imshow(a.cpu().detach().numpy(), cmap='gray')  # plt.get_cmap('gist_gray')
    plt.show()
    plt.axis('on')
    plt.close()


if __name__ == '__main__':
    pass
