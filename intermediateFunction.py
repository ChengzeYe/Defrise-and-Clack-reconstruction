import numpy as np
import torch
from pyronn.ct_reconstruction.geometry.geometry import Geometry
from pyronn.ct_reconstruction.helpers.filters.weights import cosine_weights_3d
from pyronn.ct_reconstruction.helpers.trajectories.circular_trajectory import circular_trajectory_2d
from pyronn.ct_reconstruction.layers.projection_2d import ParallelProjection2D
from weight import weight_2d


def intermediate_function(projection, geometry):
    """
        Calculate the Grangeat’s intermediate function.

        Parameters:
        - projection (array): The Cone beam projection data.
        - geometry (Geometry): A geometry object.

        Returns:
        - torch.Tensor: The Grangeat’s intermediate function.
    """
    projection = torch.tensor(projection).cuda()
    cosine = torch.tensor(cosine_weights_3d(geometry).copy()).cuda()
    projection = torch.multiply(projection, cosine)
    geom_2d = geometry_radon_2d(geometry)
    radon_2d = ParallelProjection2D()
    projection = radon_2d(projection, **geom_2d)
    projection = torch.gradient(projection, dim=2)[0]
    weight = torch.tensor(weight_2d(projection.size(), geom_2d, D=geometry.source_detector_distance).copy()).cuda()
    projection = torch.multiply(projection, weight)
    return projection


def geometry_radon_2d(geometry):
    """
        Create a Geometry object for 2d Radon Transform.

        Parameters:
        - geometry (Geometry): The original 3D Cone beam geometry configuration of the CT system.

        Returns:
        - Geometry: A new 2D geometry configuration suitable for Radon transformation.
        """
    geom_2d = Geometry()
    number_of_projections = 360
    source_detector_distance = np.sqrt(
        np.square(geometry.detector_shape[-1] * geometry.detector_spacing[-1]) + np.square(
            geometry.detector_shape[-2] * geometry.detector_spacing[-2]))
    detector_shape = np.ceil(
        np.sqrt(np.square(geometry.detector_shape[-1]) + np.square(geometry.detector_shape[-2]))).astype(int)
    detector_spacing = geometry.detector_spacing[-1]
    geom_2d.init_from_parameters(volume_shape=geometry.detector_shape, volume_spacing=geometry.detector_spacing,
                                 detector_shape=[detector_shape], detector_spacing=[detector_spacing],
                                 number_of_projections=number_of_projections, angular_range=[0, np.pi],
                                 trajectory=circular_trajectory_2d,
                                 source_isocenter_distance=source_detector_distance // 2,
                                 source_detector_distance=source_detector_distance)
    return geom_2d
