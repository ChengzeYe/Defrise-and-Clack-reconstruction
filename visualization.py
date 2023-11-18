import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from intermediateFunction import geometry_radon_2d
from pyronn.ct_reconstruction.geometry.geometry import Geometry
from pyronn.ct_reconstruction.helpers.trajectories.circular_trajectory import circular_trajectory_3d


def init_geometry():
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
    number_of_projections = 360
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

def visualization1(redundancy_weight):
    geometry = init_geometry()
    geom_2d = geometry_radon_2d(geometry)
    z1 = weight_initialization(geom_2d, D=geometry.source_detector_distance)
    z2 = redundancy_weight#[redundancy_weight.shape//2]
    height, width = z1.shape
    x = np.arange(0, width, 1)
    y = np.arange(0, height, 1)
    x, y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(12, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    surf1 = ax1.plot_surface(x, y, z1, cmap='viridis')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Intensity')
    ax1.set_title('3D Surface Plot of redundancy weight(GT)')
    cbar1 = fig.colorbar(surf1, ax=ax1, pad=0.1, aspect=10)
    cbar1.set_label('Intensity')

    ax2 = fig.add_subplot(122, projection='3d')
    surf2 = ax2.plot_surface(x, y, z2, cmap='viridis')  # 替换 surf2 变量
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Intensity')
    ax2.set_title('3D Surface Plot of redundancy weight(learned)')  # 更改标题
    cbar2 = fig.colorbar(surf2, ax=ax2, pad=0.1, aspect=10)
    cbar2.set_label('Intensity')

    plt.savefig('3D Surface Plot(learned weight).png')
    plt.show()
    #plt.clf()

    plt.plot(preprocessing(redundancy_weight[:, 93]), color='black', label='learned weight line along mu')
    plt.plot(preprocessing(z1[:, 93]), color='red', label='weight line along mu(GT)')
    plt.title('learned weight line profile')
    plt.legend()
    plt.savefig('learned weight line profile.png')
    plt.show()
    #plt.clf()


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
    output = (recon_volume - np.min(recon_volume)) / (np.max(recon_volume) - np.min(recon_volume))
    return output