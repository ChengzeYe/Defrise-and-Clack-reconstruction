import os
from natsort import natsorted
import numpy as np
from PIL import Image
import torch as torch
from torch.utils.data import Dataset
from geometry_ls import Geometry
import imageio
from arbitrary_trajectory_ls import arbitrary_projection_matrix
import matplotlib.pyplot as plt
from intermediateFunction import intermediate_function


class Projections_Dataset(Dataset):
    '''Dataset used during training of the network.'''

    def __init__(self, data_path, reco_path, objects, **kwargs):

        self.images = list()
        self.headers = list()
        self.rek_header = list()
        self.rek = list()
        self.geom = list()
        # define a sub-sampling factor in angular direction
        # (all reference reconstructions are computed with full angular resolution)
        self.angluar_sub_sampling = 1
        # select of voxels per mm in one direction (higher = larger res)
        # (all reference reconstructions are computed with 10)
        self.voxel_per_mm = 20

        self.projs_rows = 972
        self.projs_cols = 768

        for entry in os.scandir(data_path):
            self.loadProjections(entry.path)
            self.loadReconstruction(entry.path)
        print(f"initialized dataset of size {len(self.images)} ")

    def loadProjections(self, data_path):
        projs_name = 'scan_{:06}.tif'
        dark_name = 'di000000.tif'
        flat_name = ['io000000.tif', 'io000001.tif']
        # select also the orbit you want to reconstruct the data from:
        # 1 higher source position, 2 middle source position, 3 lower source position
        orbit_id = 2

        data_path_full = os.path.join(data_path, 'Projections',
                                      'tubeV{}'.format(orbit_id))

        vecs_name = 'scan_geom_corrected.geom'
        # load the numpy array describing the scan geometry from file
        vecs = np.loadtxt(os.path.join(data_path_full, vecs_name))
        # get the positions we need; there are in fact 1201, but the last and first one come from the same angle
        self.vecs = vecs[range(0, 1200, self.angluar_sub_sampling)]
        # projection file indices, we need to read in the projection in reverse order due to the portrait mode acquision
        self.projs_idx = range(1200, 0, -self.angluar_sub_sampling)
        self.n_pro = self.vecs.shape[0]
        self.init_geometry()

        # create the numpy array which will receive projection data from tiff files
        projs = np.zeros((self.n_pro, self.projs_rows, self.projs_cols), dtype=np.float32)

        # transformation to apply to each image, we need to get the image from
        # the way the scanner reads it out into to way described in the projection
        # geometry
        trafo = lambda image: np.transpose(np.flipud(image))

        # load flat-field and dark-fields
        # there are two flat-field images (taken before and after acquisition), we simply average them
        dark = trafo(imageio.imread(os.path.join(data_path_full, dark_name)))
        flat = np.zeros((2, self.projs_rows, self.projs_cols), dtype=np.float32)

        for i, fn in enumerate(flat_name):
            flat[i] = trafo(imageio.imread(os.path.join(data_path_full, fn)))
        flat = np.mean(flat, axis=0)

        # load projection data
        for i in range(self.n_pro):
            projs[i] = trafo(imageio.imread(os.path.join(data_path_full, projs_name.format(self.projs_idx[i]))))

        ### pre-process data ###########################################################
        print('pre-process data', flush=True)
        # subtract the dark field, divide by the flat field, and take the negative log to linearize the data according to the Beer-Lambert law
        projs -= dark
        projs /= (flat - dark)
        np.log(projs, out=projs)
        np.negative(projs, out=projs)
        # permute data to ASTRA convention
        #projs = np.transpose(projs, (1, 0, 2))
        projs = np.ascontiguousarray(projs)# 240, 972, 768
        projs = intermediate_function(projs, self.geom[0])
        self.images.append(projs)

    def loadReconstruction(self,data_path):
        projs_name = 'fdk_pos2_{:06}.tiff'
        data_path_full = os.path.join(data_path, 'Reconstructions')
        threeD_data = np.zeros((501, 501, 501), dtype=np.float32)
        for i in range(501):
            '''slice = imageio.imread(os.path.join(data_path_full, projs_name.format(i)))
            threeD_data[i] = (slice-np.min(slice))/(np.max(slice)-np.min(slice))'''
            threeD_data[i] = imageio.imread(os.path.join(data_path_full, projs_name.format(i)))
        threeD_data = (threeD_data - np.min(threeD_data))/(np.max(threeD_data)-np.min(threeD_data))
        self.rek.append(threeD_data.copy())  # 内存可能会出问题

    def init_geometry(self):
        geom = Geometry()
        volume_size = 501
        # size of a cubic voxel in mm
        vox_sz = 0.1#1 / self.voxel_per_mm
        volume_shape = [volume_size, volume_size, volume_size]
        geom.init_from_parameters(volume_shape=volume_shape,volume_spacing=[vox_sz, vox_sz, vox_sz],
                                detector_shape=[self.projs_rows, self.projs_cols],detector_spacing=[0.15, 0.15],
                                number_of_projections=self.n_pro,angular_range=2 * np.pi,
                                trajectory=arbitrary_projection_matrix, source_isocenter_distance=66, source_detector_distance=199,vecs=self.vecs)
        self.geom.append(geom)
    
    def __getitem__(self, index):
        """ Returns one instance of the dataset including image, target, weight and path."""

        if torch.is_tensor(index):
            index = index.tolist()

        images = self.images[index]
        return images, self.rek[index]

    def __len__(self):
        """ Returns the size of the dataset. """
        return len(self.images)

    def name(self):
        """ Returns the name of the dataset.
        """

        return 'Projection_Dataset'
