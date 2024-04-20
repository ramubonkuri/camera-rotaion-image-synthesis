import numpy as np
import open3d as o3d
import torch
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import cv2

class ExternalDepthModel:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Ensure the model is loaded correctly
        self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(self.device)
        self.model.eval()

        # Transformation for the input image to match the model's requirements
        self.transform = Compose([
            Resize((384, 384)),  # Resize to match model input
            ToTensor(),  # Convert image to tensor
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
        ])

    def depth_to_pointcloud(self, depth_map, image, intrinsic_matrix):
        # Resize depth map to match the original image dimensions
        depth_map_resized = cv2.resize(depth_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

        height, width = image.shape[:2]
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        x = (xx - intrinsic_matrix[0, 2]) * depth_map_resized / intrinsic_matrix[0, 0]
        y = (yy - intrinsic_matrix[1, 2]) * depth_map_resized / intrinsic_matrix[1, 1]
        z = depth_map_resized

        # Normalize and reshape color data
        colors = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        colors = colors.reshape(-1, 3)

        # Create 3D points
        points = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)
        point_cloud.colors = o3d.utility.Vector3dVector(colors)  # Set colors

        return point_cloud

    # def change_viewpoint(self, image, new_position):
    #     # This is a simplified intrinsic matrix assuming focal length = image width and no skew
    #     intrinsic_matrix = np.array([[image.shape[1], 0, image.shape[1]/2],
    #                                  [0, image.shape[1], image.shape[0]/2],
    #                                  [0, 0, 1]])

    #     pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    #     input_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

    #     with torch.no_grad():
    #         depth_prediction = self.model(input_tensor)
    #         depth_prediction = depth_prediction.squeeze().cpu().numpy()

    #     # Convert depth map to point cloud, ensuring to pass 'image' along with other parameters
    #     point_cloud = self.depth_to_pointcloud(depth_prediction, image, intrinsic_matrix)

    #     # Example transformation (rotation + translation). Adjust as needed.
    #     rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi/4, 0))
    #     translation_vector = np.array(new_position)

    #     # Create a 4x4 transformation matrix
    #     transformation_matrix = np.eye(4)
    #     transformation_matrix[:3, :3] = rotation_matrix
    #     transformation_matrix[:3, 3] = translation_vector

    #     # Apply the transformation
    #     point_cloud.transform(transformation_matrix)

    #     return point_cloud

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

        # Convert depth map to point cloud, ensuring to pass 'image' along with other parameters
        point_cloud = self.depth_to_pointcloud(depth_prediction, image, intrinsic_matrix)

        # Example transformation (rotation + translation). Adjust as needed.
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_xyz((0, np.pi/4, 0))
        translation_vector = np.array(new_position)

        # Create a 4x4 transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = intrinsic_matrix
 

        # Apply the transformation
        point_cloud.transform(transformation_matrix)

        # Project the point cloud back onto a 2D image plane
        height, width, _ = image.shape
        intrinsic_matrix = np.array([[width, 0, width/2, 0],
                                     [0, height, height/2, 0],
                                     [0, 0, 1, 0],
                                     [0, 0, 0, 1]])

        projected_points = point_cloud.transform(np.linalg.inv(intrinsic_matrix))
        projected_points = projected_points.points

        # projected_points = np.asarray(point_cloud.points) @ intrinsic_matrix.T
        # projected_points[:, :2] /= projected_points[:, 2:]
        # projected_points = np.round(projected_points[:, :2]).astype(int)


        # Create an empty image with the same dimensions as the input image
        output_image = np.zeros_like(image)

        # Fill in the projected points in the output image
        # for point, color in zip(projected_points, point_cloud.colors):
        #     x, y = point
        #     if 0 <= x < width and 0 <= y < height:
        #         output_image[y, x] = (color * 255).astype(int)

        for point in projected_points:
            x, y, _ = point
            x = int(x)
            y = int(y)
            if 0 <= x < width and 0 <= y < height:
                output_image[y, x] = image[y, x]

        return output_image

    def visualize_point_cloud(self, point_cloud):
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(point_cloud)

        # Configure visualization settings
        render_option = vis.get_render_option()
        render_option.light_on = True
        render_option.point_size = 2  # Increase point size for better visibility
        render_option.background_color = np.asarray([0, 0, 0])

        vis.run()  # Run the visualization loop
        vis.capture_screen_image("output.png")  # Capture the visual output
        vis.destroy_window()

