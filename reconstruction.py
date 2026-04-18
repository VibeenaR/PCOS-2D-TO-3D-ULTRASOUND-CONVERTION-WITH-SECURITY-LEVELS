import numpy as np
import open3d as o3d
import cv2

def generate_3d_volume(image_stack, output_path):
    all_points = []
    all_colors = []
    
    # Fill factor to ensure the result looks like one solid mass
    z_expansion = 2.0 
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))

    for z_idx, img in enumerate(image_stack):
        # 1. Image Pre-processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        denoised = cv2.bilateralFilter(gray, 9, 75, 75)
        enhanced = clahe.apply(denoised)

        # 2. FOLLICLE DETECTION FILTER (The 'Picker')
        # We use a binary inverse threshold to find dark circular regions (follicles)
        _, thresh = cv2.threshold(enhanced, 65, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # CRITERIA: Only use slices that have at least 3 distinct follicle candidates
        # This prevents stacking 'empty' tissue slices at the start/end of the volume
        if len(contours) < 3:
            continue

        # 3. CONSTRUCT SOLID VOXEL LAYER
        h, w = enhanced.shape
        xs, ys = np.meshgrid(np.arange(w), np.arange(h))
        
        # Create sub-layers to fill vertical gaps for a 'solid' look
        for sub_z in np.linspace(0, z_expansion, 2):
            actual_z = (z_idx * z_expansion) + sub_z
            
            # Mask out background noise (intensity-based)
            mask = enhanced > 30 
            
            points = np.stack([xs[mask], ys[mask], np.full(xs[mask].shape, actual_z)], axis=1)
            
            # 4. SATURATED HEATMAP COLORING
            colored = cv2.applyColorMap(enhanced, cv2.COLORMAP_JET)
            hsv = cv2.cvtColor(colored, cv2.COLOR_BGR2HSV)
            hsv[:,:,1] = cv2.add(hsv[:,:,1], 90) # Intense saturation for clinical clarity
            rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
            
            colors = rgb[mask].reshape(-1, 3) / 255.0
            
            all_points.append(points)
            all_colors.append(colors)

    if not all_points:
        raise ValueError("Clinical Detection Failed: No significant follicles found in stack.")

    # 5. MERGE INTO ONE UNIFIED RESULT
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
    
    # Voxel merge to close any remaining microscopic gaps
    pcd = pcd.voxel_down_sample(voxel_size=0.8)
    pcd.estimate_normals()
    
    o3d.io.write_point_cloud(output_path, pcd)
    return output_path