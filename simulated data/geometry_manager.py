
import os
from natsort import natsorted
from typing import Tuple
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from pyronn.ct_reconstruction.layers.projection_3d import ConeProjection3D
from pyronn.ct_reconstruction.geometry.geometry import Geometry
from pyronn.ct_reconstruction.helpers.phantoms import shepp_logan
from helpers.primitives_3D import visualize_grid, place_sphere, place_cylinder_with_rotation, place_ellipsoid_with_rotation, place_cube_with_rotation, place_pyramid
from pyronn.ct_reconstruction.helpers.trajectories.circular_trajectory import circular_trajectory_3d
from pyronn.ct_reconstruction.helpers.trajectories.arbitrary_trajectory import  arbitrary_projection_matrix
from helpers.estimate_volume_bounding_box import load_images, calc_binary_volume


class CircularGeometrys3D:

    def __init__(self,volume_shape, volume_spacing,detector_shape,detector_spacing,number_of_projections,angular_range,source_isocenter_distance,source_detector_distance):
        """
        Initializes a CircularGeometry object with specified imaging geometry parameters.

        Parameters:
        - volume_shape: Tuple or list specifying the shape (dimensions) of the volume to be imaged.
        - volume_spacing (mm): Tuple or list specifying the spacing between voxels in each dimension of the volume.
        - detector_shape: Tuple or list specifying the shape (dimensions) of the detector.
        - detector_spacing (mm): Tuple or list specifying the spacing between pixels on the detector.
        - number_of_projections: Integer specifying the total number of projections to be acquired.
        - angular_range: Float specifying the total angular range in radians over which projections are acquired.
        - source_isocenter_distance (mm): Float specifying the distance from the X-ray source to the rotation axis (isocenter) in mm.
        - source_detector_distance (mm): Float specifying the distance from the X-ray source to the detector in mm.
        """
        self.geometry = Geometry()
        self.geometry.init_from_parameters(volume_shape=volume_shape,volume_spacing=volume_spacing,
                                detector_shape=detector_shape,detector_spacing=detector_spacing,
                                number_of_projections=number_of_projections,angular_range=angular_range,
                                trajectory=circular_trajectory_3d, 
                                source_isocenter_distance=source_isocenter_distance, source_detector_distance=source_detector_distance)


    def generate_shepp_logan_3D_phantom(self) -> Tuple[torch.tensor,torch.tensor]:
        """
        Generates a 3D Shepp-Logan phantom and its corresponding sinogram using cone beam projection.

        The method first creates a 3D Shepp-Logan phantom based on the volume shape specified in the
        geometry attribute of the class. It then computes the sinogram by applying a forward projection
        using cone beam geometry. The projection is calculated based on the parameters defined in the
        geometry attribute, including the detector shape, spacing, and the source-detector configuration.

        Returns:
            Tuple[np.array, np.array]: A tuple containing two numpy arrays. The first array is the generated
            3D mask of the Shepp-Logan phantom, and the second array is the corresponding 3D sinogram obtained through
            the cone beam forward projection.
        """
        phantom = shepp_logan.shepp_logan_3d(self.geometry.volume_shape)
        phantom = torch.tensor(np.expand_dims(phantom, axis=0).copy(),dtype=torch.float32).cuda()
        mask = (phantom != 0).float()
        sinogram = ConeProjection3D().forward(phantom, **self.geometry).cuda()
        return mask, sinogram
    
        
    def generate_3D_primitives_phantom(self,number_of_primitives = 6)-> Tuple[torch.tensor,torch.tensor]:
        """
        Generates a 3D phantom composed of a specified number of random geometric primitives (ellipsoids,
        spheres, or cubes) and computes its sinogram using cone beam projection. The primitives are added
        to the phantom with random positions, orientations, sizes, and intensities. The method returns both
        the phantom and its sinogram as PyTorch tensors.

        Parameters:
        - number_of_primitives (int, optional): The number of geometric primitives to include in the phantom.
        Defaults to 6.

        Returns:
        - Tuple[torch.tensor, torch.tensor]: A tuple containing the 3D mask of the phantom and its corresponding sinogram,
        both as PyTorch tensors. The tensors are moved to GPU memory.
        """
        phantom = np.zeros(self.geometry.volume_shape, dtype=np.float32)

        # Create empty grid
        grid = np.zeros(self.geometry.volume_shape)
       

        for i in range(number_of_primitives):
            object_type = random.choice(["ellipsoid", "sphere", "cube","pyramid","cylinder","rectangle"])
            pos = np.random.randint(0, self.geometry.volume_shape[0], 3)
            intensitiy_value = np.random.uniform(0.4, 1.0, 1)
            print(f"{i}th Random choice was {object_type}, placed at {pos} with intensity {intensitiy_value}.")
            
            if object_type == "ellipsoid":
                ellipsoid_radii = np.random.randint(1,int(self.geometry.volume_shape[0]/5),3)  # Radii along Z, Y, X axes
                place_ellipsoid_with_rotation(grid, pos, ellipsoid_radii, value=intensitiy_value)
            elif object_type == "sphere":
                radius = np.random.randint(1,int(self.geometry.volume_shape[0]/5),1)  # Radius 
                place_sphere(grid, pos, radius, value=intensitiy_value)
            elif object_type == 'rectangle': 
                cube_size = np.random.randint(1,int(self.geometry.volume_shape[0]/5),3)  # Size of the cube
                place_cube_with_rotation(grid, pos, cube_size, value=intensitiy_value)
            elif object_type == 'cube': 
                cube_size = np.random.randint(1,int(self.geometry.volume_shape[0]/5),1)  # Size of the cube
                place_cube_with_rotation(grid, pos, (cube_size[0],cube_size[0],cube_size[0]), value=intensitiy_value)
            elif object_type == 'pyramid': 
                base_size = np.random.randint(1,int(self.geometry.volume_shape[0]/5),1)  # Length of the base's side
                height = np.random.randint(1,int(self.geometry.volume_shape[0]/5),1)  # Height of the pyramid
                # Place the pyramid in the grid
                place_pyramid(grid, pos, base_size, height, intensitiy_value)
            else: # 'cylinder'
                cylinder_height = np.random.randint(1,int(self.geometry.volume_shape[0]/5),1)
                cylinder_radius = np.random.randint(1,int(self.geometry.volume_shape[0]/5),1)
                place_cylinder_with_rotation(grid, pos, cylinder_height, cylinder_radius, value=intensitiy_value)

        phantom = torch.tensor(np.expand_dims(grid, axis=0).copy(),dtype=torch.float32).cuda()
        mask = (phantom != 0).float()
        sinogram = ConeProjection3D().forward(phantom, **self.geometry).cuda()
        visualize_grid(grid)
        return mask, sinogram
    

    
    
    
class ArbitraryGeometrys3D:

    def __init__(self,headers,volume_shape, volume_spacing,detector_shape,detector_spacing):
        """
        Initializes a Arbitrary Geometry object with specified imaging geometry parameters.

        Parameters:
        - volume_shape: Tuple or list specifying the shape (dimensions) of the volume to be imaged.
        - volume_spacing (mm): Tuple or list specifying the spacing between voxels in each dimension of the volume.
        - detector_shape: Tuple or list specifying the shape (dimensions) of the detector.
        - detector_spacing (mm): Tuple or list specifying the spacing between pixels on the detector.
        """
        self.geometry = Geometry()
        self.geometry.set_header_information(headers)
        number_of_projections = len(headers)
        source_detector_distance = headers[0].focus_detector_distance_in_mm 
        source_isocenter_distance =source_detector_distance - headers[0].focus_object_distance_in_mm
        for header in headers:
            header.number_vertical_pixels, header.number_horizontal_pixels = detector_shape
        self.geometry.init_from_parameters(volume_shape=volume_shape,volume_spacing=volume_spacing,
                                detector_shape=detector_shape,detector_spacing=detector_spacing,
                                number_of_projections=number_of_projections,angular_range=5,
                                trajectory=arbitrary_projection_matrix, 
                                source_isocenter_distance=source_isocenter_distance, source_detector_distance=source_detector_distance)
        


    def generate_shepp_logan_3D_phantom(self) -> Tuple[torch.tensor,torch.tensor]:
        """
        Generates a 3D Shepp-Logan phantom and its corresponding sinogram using cone beam projection.

        The method first creates a 3D Shepp-Logan phantom based on the volume shape specified in the
        geometry attribute of the class. It then computes the sinogram by applying a forward projection
        using cone beam geometry. The projection is calculated based on the parameters defined in the
        geometry attribute, including the detector shape, spacing, and the source-detector configuration.

        Returns:
            Tuple[np.array, np.array]: A tuple containing two numpy arrays. The first array is the generated
            3D mask of the Shepp-Logan phantom, and the second array is the corresponding 3D sinogram obtained through
            the cone beam forward projection.
        """
        phantom = shepp_logan.shepp_logan_3d(self.geometry.volume_shape)
        phantom = torch.tensor(np.expand_dims(phantom, axis=0).copy(),dtype=torch.float32).cuda()
        mask = (phantom != 0).float()
        sinogram = ConeProjection3D().forward(phantom, **self.geometry).cuda()
        return mask, sinogram
    
        
    def generate_3D_primitives_phantom(self,number_of_primitives = 6)-> Tuple[torch.tensor,torch.tensor]:
        """
        Generates a 3D phantom composed of a specified number of random geometric primitives and computes 
        its sinogram using cone beam projection. The primitives are added
        to the phantom with random positions, orientations, sizes, and intensities. The method returns both
        the phantom and its sinogram as PyTorch tensors.

        Parameters:
        - number_of_primitives (int, optional): The number of geometric primitives to include in the phantom.
        Defaults to 6.

        Returns:
        - Tuple[torch.tensor, torch.tensor]: A tuple containing the 3D mask of the phantom and its corresponding sinogram,
        both as PyTorch tensors. The tensors are moved to GPU memory.
        """
        phantom = np.zeros(self.geometry.volume_shape, dtype=np.float32)

        # Create empty grid
        grid = np.zeros(self.geometry.volume_shape)
       

        for i in range(number_of_primitives):
            object_type = random.choice(["ellipsoid", "sphere", "cube","pyramid","cylinder","rectangle"])
            pos = np.random.randint(0, self.geometry.volume_shape[0], 3)
            intensitiy_value = np.random.uniform(0.4, 1.0, 1)
            print(f"{i}th Random choice was {object_type}, placed at {pos} with intensity {intensitiy_value}.")
            
            if object_type == "ellipsoid":
                ellipsoid_radii = np.random.randint(1,int(self.geometry.volume_shape[0]/5),3)  # Radii along Z, Y, X axes
                place_ellipsoid_with_rotation(grid, pos, ellipsoid_radii, value=intensitiy_value)
            elif object_type == "sphere":
                radius = np.random.randint(1,int(self.geometry.volume_shape[0]/5),1)  # Radius 
                place_sphere(grid, pos, radius, value=intensitiy_value)
            elif object_type == 'rectangle': 
                cube_size = np.random.randint(1,int(self.geometry.volume_shape[0]/5),3)  # Size of the cube
                place_cube_with_rotation(grid, pos, cube_size, value=intensitiy_value)
            elif object_type == 'cube': 
                cube_size = np.random.randint(1,int(self.geometry.volume_shape[0]/5),1)  # Size of the cube
                place_cube_with_rotation(grid, pos, (cube_size[0],cube_size[0],cube_size[0]), value=intensitiy_value)
            elif object_type == 'pyramid': 
                base_size = np.random.randint(1,int(self.geometry.volume_shape[0]/5),1)  # Length of the base's side
                height = np.random.randint(1,int(self.geometry.volume_shape[0]/5),1)  # Height of the pyramid
                # Place the pyramid in the grid
                place_pyramid(grid, pos, base_size, height, intensitiy_value)
            else: # 'cylinder'
                cylinder_height = np.random.randint(1,int(self.geometry.volume_shape[0]/5),1)
                cylinder_radius = np.random.randint(1,int(self.geometry.volume_shape[0]/5),1)
                place_cylinder_with_rotation(grid, pos, cylinder_height, cylinder_radius, value=intensitiy_value)

        phantom = torch.tensor(np.expand_dims(grid, axis=0).copy(),dtype=torch.float32).cuda()
        mask = (phantom != 0).float()
        sinogram = ConeProjection3D().forward(phantom, **self.geometry).cuda()
        visualize_grid(grid)
        return mask, sinogram
    

class MeasuredGeometry3D:

    def __init__(self,path):
        """
        Initializes a Measured Geometry object with specified imaging geometry parameters.

        Parameters:
        - volume_shape: Tuple or list specifying the shape (dimensions) of the volume to be imaged.
        - volume_spacing (mm): Tuple or list specifying the spacing between voxels in each dimension of the volume.
        """
        headers,projections = load_images(path)
        traj_type = 'circ' if np.array_equal(np.array(headers[0].agv_source_position),np.array([0,0,0])) else 'free'
        self.geometry = Geometry()
        self.geometry.set_header_information(headers)
        number_of_projections = len(headers)
        source_detector_distance =headers[0].focus_detector_distance_in_mm  #cm
        source_isocenter_distance =source_detector_distance - (headers[0].focus_object_distance_in_mm)
        pixel_pitch_in_mm = (headers[0].detector_height_in_um/1000)/ headers[0].image_height
        detector_shape= [headers[0].number_vertical_pixels, headers[0].number_horizontal_pixels]
        volume_shape = (np.max(detector_shape),np.max(detector_shape),np.max(detector_shape))
        factor = (source_detector_distance / source_isocenter_distance)#*2
        detector_spacing =[pixel_pitch_in_mm,pixel_pitch_in_mm]
        volume_spacing =  [detector_spacing[0] / factor, detector_spacing[0] / factor, detector_spacing[1] / factor]
        if traj_type == 'free':
            self.geometry.init_from_parameters(volume_shape=volume_shape,volume_spacing=volume_spacing,
                                detector_shape=detector_shape,detector_spacing=detector_spacing,
                                number_of_projections=number_of_projections,angular_range=0,
                                trajectory=arbitrary_projection_matrix, 
                                source_isocenter_distance=source_isocenter_distance, source_detector_distance=source_detector_distance)
        else:
            angular_range = np.deg2rad(headers[0].number_projection_angles * headers[0].swing_angular_step) 
            self.geometry.init_from_parameters(volume_shape=volume_shape,volume_spacing=volume_spacing,
                                detector_shape=detector_shape,detector_spacing=[pixel_pitch_in_mm,pixel_pitch_in_mm],
                                number_of_projections=number_of_projections,angular_range=angular_range,
                                trajectory=arbitrary_projection_matrix, #changed for testing the realworld dataset
                                source_isocenter_distance=source_isocenter_distance, source_detector_distance=source_detector_distance,swap_detector_axis=True)

        self.projections = projections
        self.headers = headers

    def generate_mask_and_sinogram(self):
        mask,number_of_voxel_object = calc_binary_volume(images=self.projections,headers=self.headers,
                                  voxel_size_in_mm = self.geometry.volume_spacing,volume_voxel_count=self.geometry.volume_shape[0])
        

        
        mask = torch.tensor(mask).cuda()
        self.geometry.set_volume_shape((number_of_voxel_object+2,number_of_voxel_object+2,number_of_voxel_object+2))
        
        sinogram = self.build_sinogram(self.projections).cuda()
        
        return mask, sinogram

    def build_sinogram(self,images):
        n = len(images)
        width, height = images[0].shape
        out = np.zeros((n, width, height), dtype=np.float32)
        for i in range(0, n):
            max_value = 64000
            out[i, :, :] = -np.log(np.array(images[i], dtype=np.float32)/max_value) *10  #
            # out[i, :, :] = (max_value- np.array(images[i], dtype=np.float32))#/max_value
        return torch.tensor(np.expand_dims(out, axis=0))


    


