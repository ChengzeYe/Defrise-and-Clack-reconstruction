from weight import weight_2d
import numpy as np
import torch
from pyronn.ct_reconstruction.layers.projection_2d import ParallelProjection2D
from pyronn.ct_reconstruction.helpers.filters.weights import cosine_weights_3d
from pyronn.ct_reconstruction.geometry.geometry import Geometry
from pyronn.ct_reconstruction.helpers.trajectories.circular_trajectory import circular_trajectory_2d
import matplotlib.pyplot as plt

def intermediate_function(projection, geometry):
    projection = torch.tensor(projection)# x y plan
    '''plt.imshow(projection[0])
    plt.show()'''
    cosine = torch.tensor(cosine_weights_3d(geometry).copy())
    weighted_projection = torch.multiply(projection, cosine).cuda()
    geom_2d = geometry_radon_2d(geometry)
    radon_2d = ParallelProjection2D()
    Plan_Integral = radon_2d(weighted_projection, **geom_2d)
    '''plt.imshow(Plan_Integral[0].cpu())
    plt.show()'''
    Plan_Integral_Derivative = torch.gradient(Plan_Integral, dim=2)[0]
    '''plt.imshow(Plan_Integral_Derivative[0].cpu())
    plt.show()'''
    weight = torch.tensor(weight_2d(Plan_Integral_Derivative.size(), geom_2d, D=geometry.source_detector_distance).copy()).cuda()
    Plan_Integral_Derivative = torch.multiply(Plan_Integral_Derivative, weight)
    return Plan_Integral_Derivative#s mu plan


def geometry_radon_2d(geometry):
    geom_2d = Geometry()
    number_of_projections = 120
    source_detector_distance = np.sqrt(
        np.square(geometry.detector_shape[-1] * geometry.detector_spacing[-1]) + np.square(
            geometry.detector_shape[-2] * geometry.detector_spacing[-2]))
    geom_2d.init_from_parameters(volume_shape=geometry.detector_shape, volume_spacing=geometry.detector_spacing,
                                 detector_shape=[np.ceil(source_detector_distance).astype(int)], detector_spacing=[1],
                                 number_of_projections=number_of_projections, angular_range=[0, 2 * np.pi],
                                 trajectory=circular_trajectory_2d,
                                 source_isocenter_distance=source_detector_distance // 2,
                                 source_detector_distance=source_detector_distance)
    return geom_2d
