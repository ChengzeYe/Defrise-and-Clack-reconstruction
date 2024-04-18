import numpy as np


def weight_2d(size, geometry, D):
    """
    Calculate sinogram weights for Defrise and Clack reconstruction.

    Parameters:
    - size (tuple): The size of the sinogram obtained from 2d Radon Transform.
    - geometry (object): Geometry object containing CT scan parameters such as detector spacing and number of projections.
    - D (float): The distance from the source to the detector center.

    Returns:
    - numpy.ndarray: A 2D array of weights.
    """
    s = size[2]
    cs = -(s - 1) / 2 * geometry.detector_spacing[-1]
    sd2 = D ** 2
    w = np.zeros((geometry.number_of_projections, s), dtype=np.float32)
    for mu in range(0, geometry.number_of_projections):
        for s in range(0, size[2]):
            ds = (s * geometry.detector_spacing[-1] + cs) ** 2
            w[mu, s] = (ds + sd2) / sd2
    return np.flip(w)


def weights_3d(geometry, D):
    """
        Calculate detector weights for Defrise and Clack reconstruction.

        Parameters:
        - geometry (object): An object containing the parameters of the CT scan geometry, such as volume shape and spacing.
        - D (float): The distance from the source to the detector center.

        Returns:
        - numpy.ndarray: A 2D array of weights, where each weight corresponds to a voxel's adjusted geometric value based on its position.
        """
    cu = -(geometry.volume_shape[-1] - 1) / 2 * geometry.volume_spacing[-1]
    cv = -(geometry.volume_shape[-2] - 1) / 2 * geometry.volume_spacing[-2]
    sd2 = D ** 2

    w = np.zeros((geometry.volume_shape[-2], geometry.volume_shape[-1]), dtype=np.float32)

    for v in range(0, geometry.volume_shape[-2]):
        dv = (v * geometry.volume_spacing[-2] + cv) ** 2
        for u in range(0, geometry.volume_shape[-1]):
            du = (u * geometry.volume_spacing[-1] + cu) ** 2
            w[v, u] = dv + du + sd2

    return np.flip(w)
