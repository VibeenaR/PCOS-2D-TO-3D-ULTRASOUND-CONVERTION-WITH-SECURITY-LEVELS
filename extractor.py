import vtk
from vtk.util import numpy_support
import numpy as np
import cv2
import os

def extract_and_save_slices(file_path, session_path, num_slices=25):
    """
    Extracts a high-density image stack from VTK to simulate probe movement.
    Saves slices as a sequence for 3D reconstruction.
    """
    reader = vtk.vtkGenericDataObjectReader()
    reader.SetFileName(file_path)
    reader.Update()
    
    data = reader.GetOutput()
    dims = data.GetDimensions()
    
    # Extract raw voxel data
    scals = data.GetPointData().GetScalars()
    np_data = numpy_support.vtk_to_numpy(scals)
    
    # Reshape to (X, Y, Z) - Z is the probe movement axis
    volume = np_data.reshape(dims, order='F')
    
    # Create a subfolder for the 2D image stack
    stack_folder = os.path.join(session_path, "extracted_stack")
    os.makedirs(stack_folder, exist_ok=True)
    
    # Select slice indices evenly across the volume
    indices = np.linspace(0, dims[2]-1, num_slices).astype(int)
    
    image_stack = []
    for count, i in enumerate(indices):
        slice_2d = volume[:, :, i]
        
        # Normalize to 8-bit for clinical image processing
        rescaled = cv2.normalize(slice_2d, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Save each slice in the stack
        slice_name = f"slice_{count:03d}.png"
        cv2.imwrite(os.path.join(stack_folder, slice_name), rescaled)
        
        image_stack.append(rescaled)
        
    return np.array(image_stack)