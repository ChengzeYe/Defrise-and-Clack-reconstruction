import argparse
import os

import h5py
import random

import numpy as np
import torch
from pyronn.ct_reconstruction.geometry.geometry import Geometry
from pyronn.ct_reconstruction.helpers.trajectories.circular_trajectory import (
    circular_trajectory_3d,
)
from pyronn.ct_reconstruction.layers.projection_3d import ConeProjection3D


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="3D CT Reconstruction Dataset Generator"
    )
    parser.add_argument(
        "--num_samples", type=int, default=30, help="Number of samples to generate"
    )
    parser.add_argument(
        "--volume_size", type=int, default=501, help="Size of the volume (cubic)"
    )

    parser.add_argument(
        "--volume_spacer", type=float, default=0.1, help="Spacing of the volume"
    )

    parser.add_argument(
        "--detector_row", type=float, default=972, help="Size of the detector"
    )

    parser.add_argument(
        "--detector_col", type=float, default=768, help="Size of the detector"
    )

    parser.add_argument(
        "--detector_spacer", type=float, default=0.1496, help="Spacing of the detector"
    )

    parser.add_argument(
        "--num_projections", type=int, default=400, help="Number of projections"
    )
    parser.add_argument(
        "--angular_range",
        type=float,
        default=2 * np.pi,
        help="Angular range for the projections in radians",
    )
    parser.add_argument(
        "--sdd", type=float, default=199.006195, help="Source-Detector distance"
    )
    parser.add_argument(
        "--sid", type=float, default=66.001404, help="Source-Isocenter distance"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./simulated_dataset/",
        help="Path to save the dataset",
    )
    return parser.parse_args()


def generate_random_volume(shape):
    phantom = np.zeros(shape, dtype=np.float32)

    # Center and normalize coordinates
    coords = np.mgrid[: shape[0], : shape[1], : shape[2]]
    normalized_coords = [
        (coord - (s - 1) / 2) / ((s - 1) / 2) for coord, s in zip(coords, shape)
    ]

    for _ in range(random.randint(5, 10)):
        object_type = random.choice(["ellipsoid", "sphere", "cube"])
        pos = np.random.uniform(-0.5, 0.5, 3)
        axes = np.random.uniform(0.05, 0.2, 3)
        angles = np.random.uniform(-np.pi, np.pi, 3)

        # Compute rotated and translated coordinates
        rotated_coords = rotate_and_translate(normalized_coords, pos, angles)

        if object_type == "ellipsoid":
            object_points = (
                np.sum(
                    (rotated_coords / axes[:, np.newaxis, np.newaxis, np.newaxis]) ** 2,
                    axis=0,
                )
                <= 1
            )
        elif object_type == "sphere":
            radius = np.min(axes)
            object_points = np.sum(rotated_coords**2, axis=0) <= radius**2
        else:  # cube
            object_points = np.all(
                np.abs(rotated_coords) <= axes[:, np.newaxis, np.newaxis, np.newaxis],
                axis=0,
            )

        phantom += object_points * np.random.uniform(0, 1)

    return phantom


def rotate_and_translate(coords, pos, angles):
    xc, yc, zc = [c - p for c, p in zip(coords, pos)]

    # Rotation matrices
    Rz_phi, Ry_theta, Rz_psi = [
        rotation_matrix(angle, axis) for angle, axis in zip(angles, ["z", "y", "x"])
    ]
    R = Rz_phi @ Ry_theta @ Rz_psi

    xx, yy, zz = [R[i, 0] * xc + R[i, 1] * yc + R[i, 2] * zc for i in range(3)]
    return np.array([zz, yy, xx])


def rotation_matrix(angle, axis):
    c, s = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    elif axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    elif axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    else:
        raise ValueError("Invalid axis for rotation. Choose 'x', 'y', or 'z'.")


def save_batch(batch_data, output_path, batch_index, compression="gzip"):
    batch_file_name = f"batch_{batch_index}.h5"
    batch_file_path = os.path.join(output_path, batch_file_name)
    with h5py.File(batch_file_path, "w") as h5f:
        for i, (sinogram, volume) in enumerate(batch_data):
            h5f.create_dataset(
                f"sinogram_{i}", data=sinogram.numpy(), compression=compression
            )
            h5f.create_dataset(
                f"volume_{i}", data=volume.numpy(), compression=compression
            )
    print(f"Batch {batch_index} saved to {batch_file_path}")


def create_dataset(num_samples, volume_shape, geometry_params, batch_size, output_path):
    batch_data = []
    for index in range(num_samples):
        volume = generate_random_volume(volume_shape)
        volume_tensor = torch.tensor(
            np.expand_dims(volume, axis=0), dtype=torch.float32
        ).cuda()
        sinogram = ConeProjection3D(hardware_interp=True).forward(
            volume_tensor, **geometry_params
        )
        sinogram_tensor = sinogram.detach()

        batch_data.append((sinogram_tensor.cpu(), volume_tensor.cpu()))

        if (index + 1) % batch_size == 0 or index == num_samples - 1:
            save_batch(batch_data, output_path, index // batch_size)
            batch_data = []


def main():
    args = parse_arguments()

    output_dir = os.path.dirname(args.output_path)
    os.makedirs(output_dir, exist_ok=True)

    volume_shape = (args.volume_size, args.volume_size, args.volume_size)
    volume_spacing = (args.volume_spacer, args.volume_spacer, args.volume_spacer)

    detector_shape = (args.detector_row, args.detector_col)
    detector_spacing = (args.detector_spacer, args.detector_spacer)

    geometry = Geometry()
    geometry.init_from_parameters(
        volume_shape=volume_shape,
        volume_spacing=volume_spacing,
        detector_shape=detector_shape,
        detector_spacing=detector_spacing,
        number_of_projections=args.num_projections,
        angular_range=args.angular_range,
        trajectory=circular_trajectory_3d,
        source_isocenter_distance=args.sid,
        source_detector_distance=args.sdd,
    )

    batch_size = 1
    create_dataset(args.num_samples, volume_shape, geometry, batch_size, output_dir)


if __name__ == "__main__":
    main()
