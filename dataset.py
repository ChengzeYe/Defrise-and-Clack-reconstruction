import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from pyronn.ct_reconstruction.geometry.geometry import Geometry
from pyronn.ct_reconstruction.helpers.trajectories.circular_trajectory import circular_trajectory_3d
from intermediateFunction import intermediate_function, geometry_radon_2d


class Projections_Dataset(Dataset):
    '''Dataset used during training of the network.'''

    def __init__(self, data_path_train, **kwargs):
        self.dataset = list()
        self.geom = list()
        self.angluar_sub_sampling = 1
        self.geom.append(self.init_geometry())
        for entry in os.scandir(data_path_train):
            self.load_data(entry.path)

    def load_data(self, path):
        """
            Load volume and sinogram data from an HDF5 file and append it to the dataset.

            Parameters:
            - path (str): The file path to the HDF5 file containing the data.

            Returns:
            - tuple: A tuple containing torch tensors of the volume and sinogram.
            """
        with h5py.File(path, 'r') as file:
            sinogram = file['sinogram_0'][()]
            volume = file['volume_0'][()]
        data = torch.tensor(volume), torch.tensor(sinogram)
        self.dataset.append(data)
        return data

    def init_geometry(self):
        """
            Initialize and configure the geometry.

            This function sets up the scanning geometry parameters.

            Returns:
            - Geometry: A configured Geometry object.
        """
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
        number_of_projections = 400 // self.angluar_sub_sampling
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
        volume_sample, sinogram_sample = self.dataset[index]
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
