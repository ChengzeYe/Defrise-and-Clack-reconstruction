import numpy as np
import matplotlib.pyplot as plt


def arbitrary_projection_matrix(headers,voxel_size = [0.2,0.2], swap_detector_axis=False, **kwargs):
    #Source: Auto-calibration of cone beam geometries from arbitrary rotating markers using a vector geometry formulation of projection matrices by Graetz, Jonas
    number_of_projections = len(headers)
    # init empty
    projection_matrices = np.zeros((number_of_projections, 3, 4))
    
    # Shift into left upper corner of the detector
    detector_left_corner_trans = np.eye(3) 
    detector_left_corner_trans[0, 2] = + (float(768) - 1.) / 2.#
    detector_left_corner_trans[1, 2] = + (float(972) - 1.) / 2.
    detector_left_corner_trans[0, 0] *= 1
    detector_left_corner_trans[1, 1] *= -1
    detector_left_corner_trans[2, 2] = 1.

    for p, header in enumerate(headers):
        det_h = np.array([header[6],header[7],header[8]])
        det_v = -1 * np.array([header[9],header[10],header[11]])
        source_center_in_voxel = np.array([header[0],header[1],header[2]])   # in mm
        detector_center_in_voxel = np.array([header[3],header[4],header[5]])  # in mm

        #[H|V|d-s]
        h_v_sdd = np.column_stack((det_h, det_v, (detector_center_in_voxel - source_center_in_voxel) ))
        h_v_sdd_invers = np.linalg.inv(h_v_sdd)
        # [H|V|d-s]^-1 * -s
        back_part = h_v_sdd_invers @ (-source_center_in_voxel)
        proj_matrix = np.column_stack((h_v_sdd_invers,back_part))
        projection_matrices[p] =  detector_left_corner_trans @ proj_matrix
        
        # post processing to get the same oriented outputvolume like ezrt commandline reco:
        # flip Z-Axis: Z = -Z
        projection_matrices[p][0:3, 2] = projection_matrices[p][0:3, 2] * -1.0

        # change orientation of current matrix from XYZ to YXZ: exchange the first two columns
        projection_matrices[p][0:3, 0:2] = np.flip(projection_matrices[p][0:3, 0:2], axis=1)
    return projection_matrices

