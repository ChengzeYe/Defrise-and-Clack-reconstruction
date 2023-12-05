from weight import weight_2d
import numpy as np
import torch
from pyronn.ct_reconstruction.layers.projection_2d import ParallelProjection2D
from pyronn.ct_reconstruction.helpers.filters.weights import cosine_weights_3d
from pyronn.ct_reconstruction.geometry.geometry import Geometry
from pyronn.ct_reconstruction.helpers.trajectories.circular_trajectory import circular_trajectory_2d
import matplotlib.pyplot as plt
import torch.nn.functional as F

def intermediate_function(projection, geometry):
    projection = torch.tensor(projection).cuda()# x y plan
    '''plt.imshow(projection[0].cpu())
    plt.show()'''
    cosine = torch.tensor(cosine_weights_3d(geometry).copy()).cuda()
    weighted_projection = torch.multiply(projection, cosine)
    geom_2d = geometry_radon_2d(geometry)
    radon_2d = ParallelProjection2D()
    Plan_Integral = radon_2d(weighted_projection, **geom_2d)
    '''plt.imshow(Plan_Integral[0].cpu())
    plt.show()'''

    Plan_Integral_Derivative = torch.gradient(Plan_Integral, dim=2)[0]

    '''Plan_Integral = torch.unsqueeze(Plan_Integral, dim=1)
    sobel = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).cuda()
    Plan_Integral_Derivative = F.conv2d(Plan_Integral, sobel.view(1, 1, 3, 3), stride=1, padding=1)
    Plan_Integral_Derivative = torch.squeeze(Plan_Integral_Derivative, dim=1)'''

    '''plt.imshow(Plan_Integral_Derivative[0].cpu())
    plt.show()'''

    weight = torch.tensor(weight_2d(Plan_Integral_Derivative.size(), geom_2d, D=geometry.source_detector_distance).copy()).cuda()
    Plan_Integral_Derivative = torch.multiply(Plan_Integral_Derivative, weight)

    '''plt.imshow(Plan_Integral_Derivative[0].cpu())
    plt.show()'''
    return Plan_Integral_Derivative# mu s plan


def geometry_radon_2d(geometry):
    geom_2d = Geometry()
    number_of_projections = geometry.number_of_projections
    source_detector_distance = np.sqrt(
        np.square(geometry.detector_shape[-1] * geometry.detector_spacing[-1]) + np.square(
            geometry.detector_shape[-2] * geometry.detector_spacing[-2]))
    detector_shape = np.ceil(np.sqrt(np.square(geometry.detector_shape[-1]) + np.square(geometry.detector_shape[-2]))).astype(int)
    detector_spacing = geometry.detector_spacing[-1]
    geom_2d.init_from_parameters(volume_shape=geometry.detector_shape, volume_spacing=geometry.detector_spacing,
                                 detector_shape=[detector_shape], detector_spacing=[detector_spacing],#0.7
                                 number_of_projections=number_of_projections, angular_range=[0, 2 * np.pi],
                                 trajectory=circular_trajectory_2d,
                                 source_isocenter_distance=source_detector_distance // 2,
                                 source_detector_distance=source_detector_distance)
    return geom_2d
