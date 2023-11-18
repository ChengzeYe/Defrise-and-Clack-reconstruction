import os
import numpy as np
import torch as torch
from torch.utils.data import Dataset
#from geometry_ls import Geometry
from pyronn.ct_reconstruction.geometry.geometry import Geometry
from pyronn.ct_reconstruction.helpers.trajectories.circular_trajectory import circular_trajectory_3d
#import matplotlib.pyplot as plt
from intermediateFunction import intermediate_function, geometry_radon_2d


class Projections_Dataset(Dataset):
    '''Dataset used during training of the network.'''

    def __init__(self, data_path, reco_path, objects, **kwargs):
        self.dataset = list()
        self.geom = list()
        self.angluar_sub_sampling = 1
        self.geom.append(self.init_geometry())
        for entry in os.scandir(data_path):
            self.dataset.extend(torch.load(entry.path)[0:2])

#  一个长度为十的list 每个元素是个元组，元素1是 1， 501，501，501  元素二是1，360，972 768 别忘记翻转xy轴

    def init_geometry(self):
        voxel_per_mm = 10
        projs_rows = 972
        projs_cols = 768

        volume_size = 501
        volume_shape = (volume_size, volume_size, volume_size)
        volume_spacing = (1 / voxel_per_mm, 1 / voxel_per_mm, 1 / voxel_per_mm)

        # Detector Parameters:
        detector_shape = (projs_rows, projs_cols)
        detector_spacing = (0.1496, 0.1496)

        # Trajectory Parameters:
        number_of_projections = 360//self.angluar_sub_sampling
        angular_range = 2 * np.pi

        sdd = 199.006195
        sid = 66.001404

        # create Geometry class
        geometry = Geometry()
        geometry.init_from_parameters(
            volume_shape=volume_shape,
            volume_spacing=volume_spacing,
            detector_shape=detector_shape,
            detector_spacing=detector_spacing,
            number_of_projections=number_of_projections,
            angular_range=angular_range,
            trajectory=circular_trajectory_3d,
            source_isocenter_distance=sid,
            source_detector_distance=sdd, swap_detector_axis=False)
        return geometry

    def __getitem__(self, index):
        """ Returns one instance of the dataset including image, target, weight and path."""
        #sample_index = np.random.randint(len(self.dataset))
        sample_index = 1
        volume_sample, sinogram_sample = self.dataset[sample_index]
        volume_sample = torch.squeeze(volume_sample, dim=0).cpu().numpy()
        sinogram_sample = torch.squeeze(sinogram_sample, dim=0)[:: self.angluar_sub_sampling, :, :].cpu().numpy()
        return intermediate_function(sinogram_sample, self.geom[0]), volume_sample

    def __len__(self):
        """ Returns the size of the dataset. """
        return len(self.dataset)

    def name(self):
        """ Returns the name of the dataset.
        """

        return 'Projection_Dataset'