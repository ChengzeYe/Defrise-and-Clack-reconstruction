# Copyright [2019] [Christopher Syben, Markus Michen]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import copy
from io import FileIO
import numpy as np
import torch
from typing import Callable, Tuple
import warnings
from pyronn.ct_reconstruction.helpers.trajectories.circular_trajectory import circular_trajectory_3d
# from pyronn.ct_reconstruction.helpers.trajectories.arbitrary_trajectory import arbitrary_projection_matrix
from arbitrary_trajectory_Walnut import arbitrary_projection_matrix
from importlib.util import find_spec

if find_spec('PythonTools'):
    from PythonTools import ezrt_header

'''
try:
    from PythonTools import ezrt_header
except ImportError as e:
    pass
'''


class Geometry:
    """
        The Base Class for the different Geometries. Provides commonly used members.
    """

    def __init__(self):
        """
            Constructor of Geometry Class.
        Args:
            volume_shape:               The volume size in Z, Y, X order.
            volume_spacing:             The spacing between voxels in Z, Y, X order.
            detector_shape:             Shape of the detector in Y, X order.
            detector_spacing:           The spacing between detector voxels in Y, X order.
            number_of_projections:      Number of equidistant projections.
            angular_range:              The covered angular range.
            trajectory->array:          The function pointer which computes the trajectory.
            source_detector_distance:   The source to detector distance (sdd). Only for fan and cone-beam geometries. Default = 0.
            source_isocenter_distance:  The source to isocenter distance (sid). Only for fan and cone-beam geometries. Default = 0.
            trajectory:Callable         Function pointer to the trajectory compute function.
            projection_multiplier:      Constant factor of the distance weighting for FDK Cone-beam reconstruction. Is computed automatically from geometry.
            step_size:                  Step size for sampling points along the ray in the Cone-beam projector. Default=0.2* voxel_spacing
            swap_detector_axis:         2D Detector axis direction. Default is [1,0,0], swaped [-1,0,0]

        """
        self.gpu_device = True
        self.traj_func = circular_trajectory_3d
        self.np_dtype = np.float32  # datatype for np.arrays make sure everything will be float32
        self.parameter_dict = {}
        self.headers = None
        # Needed parameters
        self.parameter_dict.setdefault('volume_shape', None)
        self.parameter_dict.setdefault('volume_spacing', None)
        self.parameter_dict.setdefault('detector_shape', None)
        self.parameter_dict.setdefault('detector_spacing', None)
        self.parameter_dict.setdefault('number_of_projections', None)
        # self.parameter_dict.setdefault('angular_range', None)
        self.parameter_dict.setdefault('trajectory', None)
        # Optional parameters, neccessary for fan- and cone-beam geometry
        self.parameter_dict.setdefault('source_detector_distance', None)
        self.parameter_dict.setdefault('source_isocenter_distance', None)
        # Optional paramters, neccessarry for cone-beam geometry
        self.parameter_dict.setdefault('projection_multiplier', None)
        self.parameter_dict.setdefault('step_size', None)
        self.parameter_dict.setdefault('swap_detector_axis', False)

    def init_from_parameters(self,vecs, volume_shape: Tuple[int, ...], volume_spacing: Tuple[float, ...],
                             detector_shape: Tuple[int, ...], detector_spacing: Tuple[float, ...],
                             number_of_projections: int, angular_range: Tuple[float, ...], trajectory: Callable,
                             source_detector_distance: float = .0, source_isocenter_distance: float = .0,
                             swap_detector_axis: bool = False) -> None:
        self.parameter_dict['swap_detector_axis'] = swap_detector_axis
        # Volume Parameters:
        self.parameter_dict['volume_shape'] = np.array(volume_shape)
        self.parameter_dict['volume_spacing'] = np.array(volume_spacing, dtype=self.np_dtype)
        self.parameter_dict['volume_origin'] = -(self.parameter_dict['volume_shape'] - 1) / 2.0 * self.parameter_dict['volume_spacing']

        # Detector Parameters:
        self.parameter_dict['detector_shape'] = np.array(detector_shape)
        self.parameter_dict['detector_spacing'] = np.array(detector_spacing, dtype=self.np_dtype)
        self.parameter_dict['detector_origin'] = -(self.parameter_dict['detector_shape'] - 1) / 2.0 * \
                                                 self.parameter_dict['detector_spacing']

        # Trajectory Parameters:
        self.parameter_dict['number_of_projections'] = number_of_projections
        if isinstance(angular_range, list):
            self.parameter_dict['angular_range'] = angular_range
        else:
            self.parameter_dict['angular_range'] = [0, angular_range]
        self.parameter_dict['sinogram_shape'] = np.array(
            [self.parameter_dict['number_of_projections'], *self.parameter_dict['detector_shape']])
        self.parameter_dict['source_detector_distance'] = source_detector_distance
        self.parameter_dict['source_isocenter_distance'] = source_isocenter_distance
        self.parameter_dict['trajectory'] = trajectory(headers=vecs,**self.parameter_dict)
        self.traj_func = trajectory

        # Containing the constant part of the distance weight and discretization invariant
        self.parameter_dict['projection_multiplier'] = self.parameter_dict['source_isocenter_distance'] * \
                                                       self.parameter_dict['source_detector_distance'] * \
                                                       detector_spacing[-1] * np.pi / self.parameter_dict[
                                                           'number_of_projections']
        self.parameter_dict['step_size'] = 0.2

    def to_json(self, path: str) -> None:
        """
            Saves the geometry as json in the file denotes by @path
            The internal parameter_dict is stored including the trajectory array, hence, changes in the json file are not validated if correct or not.
            TODO: Future feature should be to store only the core parameters and recompute the origins, trajectory, etc. from core parameters when loaded via from_json() method.
        """
        import json
        class NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)

        try:
            with open(path, 'w+') as outfile:
                json.dump(self.parameter_dict, outfile, cls=NumpyEncoder)
        except FileNotFoundError and FileExistsError:
            print('Error while saving geometry to file.')

    @staticmethod
    def from_json(path: str):
        import json
        loaded_geom = Geometry()
        try:
            with open(path, 'r') as infile:
                loaded_geom.parameter_dict = json.load(infile)
        except FileNotFoundError:
            print('Error while loading geometry from file.')
        for key, value in loaded_geom.parameter_dict.items():
            if isinstance(value, list):
                loaded_geom.parameter_dict[key] = np.asarray(value)

        return loaded_geom

    def init_from_EZRT_header(self, projection_headers: Tuple[str, ...], reco_header=None, detector_shape=None,
                              detector_spacing=None, swap_detector_axis: bool = True, **kwargs) -> None:
        self.headers = projection_headers
        header = self.headers[0]
        traj_type = 'circ' if np.array_equal(np.array(header.agv_source_position), np.array([0, 0, 0])) else 'free'
        self.traj_func = arbitrary_projection_matrix
        print(traj_type)
        if traj_type == 'circ':
            angular_range = 2 * np.pi  # full circle
            self.parameter_dict['angular_range'] = [0, angular_range]

        # if reco_header.num_voxel_z != 0 and reco_header.num_voxel_y != 0 and reco_header.num_voxel_x != 0:
        #     self.parameter_dict['volume_shape'] = np.asarray(
        #         [reco_header.num_voxel_z, reco_header.num_voxel_y, reco_header.num_voxel_x])
        # else:
        #     warnings.warn("Warning: No valid volume shape for Reconstruction could be defined in the geometry")

        # if reco_header.voxel_size_z_in_um != 0 and reco_header.voxel_size_x_in_um != 0:
        #     self.parameter_dict['volume_spacing'] = np.asarray(
        #         [reco_header.voxel_size_z_in_um, reco_header.voxel_size_x_in_um, reco_header.voxel_size_x_in_um]) / 1000.0
        # elif reco_header.voxel_size_in_um != 0:
        #     self.parameter_dict['volume_spacing'] = np.full(shape=3, fill_value=header.voxel_size_in_um / 1000.0,
        #                                                     dtype=self.np_dtype)
        # else:
        #     warnings.warn("Warning: No valid volume spacing for Reconstruction could be defined in the geometry")
        scaling_factor = header._focus_detector_distance_in_um / header._focus_object_distance_in_um
        if reco_header == None:
            # Berechnung der Größen für die Rekonstruktion anhand der Header Daten
            pixel = max([header.number_horizontal_pixels, header.number_vertical_pixels])
            voxelcount = np.ceil(pixel * scaling_factor)
            # Test if voxelcount is dividable by 8
            if voxelcount % 8 != 0:
                voxelcount = np.ceil(voxelcount / 8) * 8
            self.parameter_dict['volume_shape'] = np.asarray(
                [voxelcount, voxelcount, voxelcount])
            self.parameter_dict['volume_spacing'] = np.asarray(
                [header.detector_width_in_um / pixel, header.detector_width_in_um / pixel,
                 header.detector_width_in_um / pixel]) / 1000.0
        elif reco_header.num_voxel_z != 0 and reco_header.num_voxel_y != 0 and reco_header.num_voxel_x != 0:
            self.parameter_dict['volume_shape'] = np.asarray(
                [reco_header.num_voxel_z, reco_header.num_voxel_y, reco_header.num_voxel_x])
            if reco_header.voxel_size_z_in_um != 0 and reco_header.voxel_size_x_in_um != 0:
                self.parameter_dict['volume_spacing'] = np.asarray(
                    [reco_header.voxel_size_z_in_um, reco_header.voxel_size_x_in_um,
                     reco_header.voxel_size_x_in_um]) / (1000.0 / (scaling_factor * 1.11))
            elif reco_header.voxel_size_in_um != 0:
                self.parameter_dict['volume_spacing'] = np.full(shape=3, fill_value=header.voxel_size_in_um / 1000.0,
                                                                dtype=self.np_dtype)
        else:
            warnings.warn(
                "Warning: No valid volume shape and/or volume spacing for Reconstruction could be defined in the geometry")

        if self.parameter_dict['volume_shape'] is None or self.parameter_dict['volume_spacing'] is None:
            warnings.warn("Warning: No valid volume origin for Reconstruction could be computed")
        else:
            self.parameter_dict['volume_origin'] = -(self.parameter_dict['volume_shape'] - 1) / 2.0 * \
                                                   self.parameter_dict['volume_spacing']

        # Detector Parameters:
        if detector_shape == None:
            self.parameter_dict['detector_shape'] = np.array(
                [header.number_vertical_pixels, header.number_horizontal_pixels])
            self.parameter_dict['detector_spacing'] = np.full(shape=2, fill_value=(
                                                                                              header.detector_height_in_um / 1000.0) / header.number_vertical_pixels,
                                                              dtype=self.np_dtype)  # np.array(header.pixel_width_in_um, dtype=self.np_dtype)
        else:
            self.parameter_dict['detector_shape'] = np.array(detector_shape)
            self.parameter_dict['detector_spacing'] = np.full(shape=2, fill_value=detector_spacing,
                                                              dtype=self.np_dtype)  # np.array(header.pixel_width_in_um, dtype=self.np_dtype)

        self.parameter_dict['detector_origin'] = -(self.parameter_dict['detector_shape'] - 1) / 2.0 * \
                                                 self.parameter_dict['detector_spacing']

        # Trajectory Parameters:
        self.parameter_dict['number_of_projections'] = len(projection_headers)  #-270#___________________________________________________________________________

        self.parameter_dict['sinogram_shape'] = np.array(
            [self.parameter_dict['number_of_projections'], *self.parameter_dict['detector_shape']])
        self.parameter_dict['source_detector_distance'] = header.focus_detector_distance_in_mm
        self.parameter_dict['source_isocenter_distance'] = header.focus_object_distance_in_mm
        self.__generate_trajectory__()

        # Containing the constant part of the distance weight and discretization invarian
        self.parameter_dict['projection_multiplier'] = self.parameter_dict['source_isocenter_distance'] * \
                                                       self.parameter_dict['source_detector_distance'] * \
                                                       self.parameter_dict['detector_spacing'][-1] * np.pi / \
                                                       self.parameter_dict['number_of_projections']

        self.parameter_dict['step_size'] = 1.0
        self.parameter_dict['swap_detector_axis'] = swap_detector_axis

    def cuda(self) -> None:
        self.gpu_device = True

    def cpu(self) -> None:
        self.gpu_device = False

    def keys(self) -> str:
        return self.parameter_dict

    def __getitem__(self, key: str) -> torch.Tensor:
        try:
            parameter = self.parameter_dict[key]
            if hasattr(parameter, '__len__'):
                tmp_tensor = torch.Tensor(parameter)
            else:
                tmp_tensor = torch.Tensor([parameter])
        except:
            print('Attribute <' + key + '> could not be transformed to torch.Tensor')
        if self.gpu_device:
            return tmp_tensor.cuda()
        else:
            return tmp_tensor.cpu()

    def __generate_trajectory__(self) -> None:
        if self.traj_func == circular_trajectory_3d:
            self.parameter_dict['trajectory'] = self.traj_func(**self.parameter_dict)
        else:
            self.parameter_dict['trajectory'] = self.traj_func(self.headers,
                                                               voxel_size=self.parameter_dict['volume_spacing'],
                                                               swap_detector_axis=self.parameter_dict[
                                                                   'swap_detector_axis'])

    def fan_angle(self) -> float:
        return np.arctan(
            ((self.parameter_dict['detector_shape'][-1] - 1) / 2.0 * self.parameter_dict['detector_spacing'][-1]) /
            self.parameter_dict['source_detector_distance'])

    def cone_angle(self) -> float:
        return np.arctan(
            ((self.parameter_dict['detector_shape'][-2] - 1) / 2.0 * self.parameter_dict['detector_spacing'][-2]) /
            self.parameter_dict['source_detector_distance'])

    def set_detector_shift(self, detector_shift: Tuple[float, ...]) -> None:
        """
            Applies a detector shift in px to the geometry.
            This triggers a recomputation of the trajectory. Projection matrices will be overwritten.

            :param detector_shift: Tuple[float,...] with [y,x] convention in Pixels
        """
        # change the origin according to the shift
        self.parameter_dict['detector_origin'] = self.parameter_dict['detector_origin'] + (
                detector_shift * self.parameter_dict['detector_spacing'])
        # recompute the trajectory with the applied shift
        # TODO: For now cone-beam circular full scan is fixed as initialization. Need to be reworked such that header defines the traj function
        self.__generate_trajectory__()

    def set_volume_slice(self, slice):
        """
        select one slice reconstruction geometry from the 3d objection, this function will correct reconstructions far from center.

        :param slice: selected slice
        :return: a new geometry for selected slice
        """
        # volume_shift = slice - (self.parameter_dict['volume_shape'][0] - 1) / 2.0
        geo = copy.deepcopy(self)
        geo.parameter_dict['volume_origin'][0] = geo.parameter_dict['volume_origin'][0] + (
                slice * geo.parameter_dict['volume_spacing'][0])
        geo.parameter_dict['volume_shape'][0] = 1
        geo.__generate_trajectory__()
        return geo

    def set_angle_range(self, angle_range):
        """
        change the range of angle(geometry). WARNING: this will change the original geometry

        :param angle_range: list or single value, if a single value is given, angle_range will be [0, value]
        :return: None
        """
        if isinstance(angle_range, list):
            self.parameter_dict['angular_range'] = angle_range
        else:
            self.parameter_dict['angular_range'] = [0, angle_range]
        self.__generate_trajectory__()

    def swap_axis(self, swap_det_axis: bool) -> None:
        """
            Sets the direction of the rotatation of the system.
            This triggers a recomputation of the trajectory. Projection matrices will be overwritten.

            :param counter_clockwise: wether the system rotates counter clockwise (True) or not.
        """
        self.parameter_dict['swap_det_axis'] = swap_det_axis
        self.__generate_trajectory__()

    def slice_the_geometry(self, slices):
        """
        Create several(slices) sub-geometries to overcome memory not enough.

        :param: slices: the amount of sub-geometries
        :return: a list of sub-geometries
        """
        nop = self.number_of_projections // slices
        angular_inc = 2 * np.pi / slices
        sliced_geos = []

        for i in range(slices):
            geo = copy.deepcopy(self)
            start_angle = 2 * i * np.pi / slices
            geo.number_of_projections = self.number_of_projections // slices
            geo.set_angle_range([start_angle, start_angle + angular_inc])
            sliced_geos.append(geo)
        return sliced_geos

    @property
    def volume_shape(self) -> Tuple[int, ...]:
        return tuple(self.parameter_dict['volume_shape'])

    @property
    def volume_spacing(self) -> Tuple[float, ...]:
        return tuple(self.parameter_dict['volume_spacing'])

    @property
    def detector_shape(self) -> Tuple[int, ...]:
        return tuple(self.parameter_dict['detector_shape'])


    def sinogram_shape(self, value):
        self.parameter_dict['sinogram_shape'] = value


    @property
    def detector_spacing(self) -> Tuple[float, ...]:
        return tuple(self.parameter_dict['detector_spacing'])

    @property
    def number_of_projections(self) -> int:
        return int(self.parameter_dict['number_of_projections'])

    @number_of_projections.setter
    def number_of_projections(self, value):
        self.parameter_dict['number_of_projections'] = value

    @property
    def angular_range(self) -> float:
        return self.parameter_dict['angular_range']

    @property
    def trajectory(self) -> Tuple[float, ...]:
        return self.parameter_dict['trajectory']

    @trajectory.setter
    def trajectory(self, value):
        self.parameter_dict['trajectory'] = value

    @property
    def source_detector_distance(self) -> float:
        return self.parameter_dict['source_detector_distance']

    @property
    def source_isocenter_distance(self) -> float:
        return self.parameter_dict['source_isocenter_distance']

    @property
    def projection_multiplier(self) -> float:
        return self.parameter_dict['projection_multiplier']

    @property
    def step_size(self) -> float:
        return self.parameter_dict['step_size']

    @property
    def swap_detector_axis(self) -> bool:
        return self.parameter_dict['swap_detector_axis']

    @property
    def is_gpu(self) -> bool:
        return self.gpu_device