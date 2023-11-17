import os
from natsort import natsorted
import numpy as np
from PIL import Image
import torch as torch
from torch.utils.data import Dataset
from geometry_ls import Geometry
import imageio
from arbitrary_trajectory_Walnut import arbitrary_projection_matrix
import matplotlib.pyplot as plt
from intermediateFunction import intermediate_function, geometry_radon_2d
import cupy as cp


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
        self.angluar_sub_sampling = 12
        # select of voxels per mm in one direction (higher = larger res)
        # (all reference reconstructions are computed with 10)
        self.voxel_per_mm = 20
        self.projs_rows = 972
        self.projs_cols = 768
        self.geom_2d = None
        data_path = reco_path
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
        #redundancy_weight(self.vecs, self.geom_2d)
        #intermediate_variable(self.vecs, self.geom_2d)
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
                                detector_shape=[self.projs_rows, self.projs_cols],detector_spacing=[0.1496, 0.1496],
                                number_of_projections=self.n_pro,angular_range=2 * np.pi,
                                trajectory=arbitrary_projection_matrix, source_isocenter_distance=66.001404, source_detector_distance=199.006195, vecs=self.vecs)
        self.geom.append(geom)
        self.geom_2d = geometry_radon_2d(geom)
    
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






def intermediate_variable(vecs, geom_2d):
    l = np.zeros((vecs.shape[0],geom_2d.number_of_projections,geom_2d.detector_shape[-1]))
    ll = np.zeros((vecs.shape[0], geom_2d.number_of_projections, geom_2d.detector_shape[-1]))
    lll = np.zeros((vecs.shape[0],geom_2d.number_of_projections,geom_2d.detector_shape[-1]))
    angular_increment = 2 * np.pi / geom_2d.number_of_projections
    gradient_vecs = np.gradient(vecs, axis=0)
    for p, vec in enumerate(vecs):
        print(p)
        det_h = np.array([vec[6], vec[7], vec[8]])
        det_v = np.array([vec[9], vec[10], vec[11]])
        source_center_in_voxel = np.array([vec[0], vec[1], vec[2]])
        detector_center_in_voxel = np.array([vec[3], vec[4], vec[5]])
        axis_align_R = np.eye(3, 3)
        axis_align_R[0:3, 0] = det_h
        axis_align_R[0:3, 1] = det_v
        axis_align_R[0:3, 2] = np.cross(det_h, det_v)
        axis_align_R = np.linalg.pinv(axis_align_R)
        gradient_vec = np.dot(axis_align_R, gradient_vecs[p, 0:3])
        gradient_vec = gradient_vec/np.linalg.norm(gradient_vec)
        #translation = np.array([0, np.linalg.norm(detector_center_in_voxel), 0])
        vec = np.dot(axis_align_R, vecs[p, 0:3] - detector_center_in_voxel)
        vec = vec/np.linalg.norm(vec)
        s = geom_2d.detector_shape[-1]
        cs = -(s - 1) / 2 * geom_2d.detector_spacing[-1]
        D = np.linalg.norm(source_center_in_voxel)+np.linalg.norm(detector_center_in_voxel)
        sd2 = D ** 2
        for mu in range(0, geom_2d.number_of_projections):
            for s in range(0, geom_2d.detector_shape[-1]):
                ds = (s * geom_2d.detector_spacing[-1] + cs) ** 2
                theta = np.array([D*np.cos(mu*angular_increment-np.pi/2), D*np.sin(mu*angular_increment-np.pi/2), s * geom_2d.detector_spacing[-1] + cs])/np.sqrt(sd2+ds)
                #theta = np.array([D * np.cos(mu * angular_increment), D * np.sin(mu * angular_increment), s])/np.sqrt(sd2+ds)
                l[p][mu, s] = np.dot(vec, theta)
                ll[p][mu, s] = np.dot(gradient_vec, theta)
                lll[p][mu, s] = np.dot(gradient_vec, theta)/np.sqrt(sd2+ds)
    return np.abs(l), np.abs(ll), np.abs(lll)


def redundancy_weight(vecs, geom_2d):
    a = -1/(4*cp.pi**2)
    l, ll, lll = intermediate_variable(vecs, geom_2d)
    l = cp.array(l)
    ll = cp.array(ll)
    lll = cp.array(lll)
    denominator = np.zeros_like(ll)
    for i in range(l.shape[0]):
        print(i)
        for j in range(l.shape[1]):
            for k in range(l.shape[2]):
                current_value = l[i, j, k]
                if ll[i, j, k] == 0 or l[i, j, k] == 0:
                    denominator[i, j, k] = 1
                    continue
                equal_positions = cp.where(cp.isclose(l, current_value, atol=1.e-10, rtol=1.e-10))
                extracted_values = ll[equal_positions[0], equal_positions[1], equal_positions[2]]**2
                denominator[i, j, k] = cp.sum(extracted_values)-ll[i, j, k]**2
    m = ll**2/denominator
    redundancy_weight = a*m*lll
    np.save('redundancy_weight_120_100_circular', redundancy_weight)
    np.save('m_120_100_circular', m)
    print("saved")
    print(np.average(np.array(b)))
    return redundancy_weight


# l : /a(lamda).theta/
# ll: /a'(lamda).theta/
# lll: /a'(lamda).theta//sqrt s2+D2