import torch
import numpy as np
import cv2
import open3d as o3d
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

class ExternalDepthModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(self.device)
        self.model.eval()

        self.transform = Compose([
            Resize((384, 384)),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def depth_to_pointcloud(self, depth_map, intrinsic_matrix):
        height, width = depth_map.shape
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        x = (xx - intrinsic_matrix[0, 2]) * depth_map / intrinsic_matrix[0, 0]
        y = (yy - intrinsic_matrix[1, 2]) * depth_map / intrinsic_matrix[1, 1]
        z = depth_map
        points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        return point_cloud

    def change_viewpoint(self, image, new_position):
        # This is a simplified intrinsic matrix assuming focal length = image width and no skew
        intrinsic_matrix = np.array([[image.shape[1], 0, image.shape[1]/2],
                                     [0, image.shape[1], image.shape[0]/2],
                                     [0, 0, 1]]) 
        
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            depth_prediction = self.model(input_tensor)
            depth_prediction = depth_prediction.squeeze().cpu().numpy()
            
        # Convert depth map to point cloud
        point_cloud = self.depth_to_pointcloud(depth_prediction, intrinsic_matrix)
        
        # Example transformation (rotation + translation). Adjust as needed.
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi/4, 0))
        translation_vector = np.array(new_position)
        
        # Create a 4x4 transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = rotation_matrix
        transformation_matrix[:3, 3] = translation_vector
        
        # Apply the transformation
        point_cloud.transform(transformation_matrix)

        # Synthesize the new view
        # synthesized_image = self.synthesize_new_view(image, point_cloud, intrinsic_matrix, transformation_matrix)
        # return synthesized_image
        
        return point_cloud
    def synthesize_new_view(self, image, point_cloud, intrinsic_matrix, transformation_matrix):
        # Convert the point cloud to a numpy array
        points = np.asarray(point_cloud.points)

        # Apply the transformation matrix to the points
        transformed_points = np.hstack((points, np.ones((points.shape[0], 1))))
        transformed_points = (transformation_matrix @ transformed_points.T).T

        # Project the 3D points to the 2D image plane
        projected_points = (intrinsic_matrix @ transformed_points[:, :3].T).T
        projected_points[:, 0] /= projected_points[:, 2]
        projected_points[:, 1] /= projected_points[:, 2]
        projected_points = projected_points.astype(np.int32)

        # Create a new image with the same size as the input image
        height, width, _ = image.shape
        synthesized_image = np.zeros_like(image)

        # Iterate over the projected 2D points and copy the corresponding pixel values from the input image
        for point in projected_points:
            x, y = point[:2]
            if 0 <= x < width and 0 <= y < height:
                synthesized_image[y, x] = image[y, x]


        return synthesized_image

    def visualize_point_cloud(self, point_cloud):
        o3d.visualization.draw_geometries([point_cloud])
