import numpy as np
from geometry_ls import Geometry


def weight_2d(size, geometry, D):
    s = size[2]
    cs = -(s - 1) / 2 * geometry.detector_spacing[-1]
    sd2 = D ** 2
    w = np.zeros((geometry.number_of_projections, s), dtype=np.float32)
    for mu in range(0, geometry.number_of_projections):
        for s in range(0, size[2]):
            ds = (s * geometry.detector_spacing[-1] + cs) ** 2
            w[mu, s] = (ds+sd2) /sd2
    return np.flip(w)


def weights_3d(geometry, D):
    cu = -(geometry.volume_shape[-1] - 1) / 2 * geometry.volume_spacing[-1]
    cv = -(geometry.volume_shape[-2] - 1) / 2 * geometry.volume_spacing[-2]
    sd2 = D ** 2

    w = np.zeros((geometry.volume_shape[-2], geometry.volume_shape[-1]), dtype=np.float32)

    for v in range(0, geometry.volume_shape[-2]):
        dv = (v * geometry.volume_spacing[-2] + cv) ** 2
        for u in range(0, geometry.volume_shape[-1]):
            du = (u * geometry.volume_spacing[-1] + cu) ** 2
            w[v, u] = dv + du +sd2

    return np.flip(w)


