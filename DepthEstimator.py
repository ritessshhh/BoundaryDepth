from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import torch
from PIL import Image
import open3d as o3d
from tsr.system import TSR
from tsr.utils import resize_foreground
import json


class BoundaryDepthExtractor:
    def __init__(self, default_zoom=0.5, default_up_vector=[0, 1, 0], default_front_vector=[0, 0, 1]):
        # Initialize the Depth Estimator
        self.default_zoom = default_zoom
        self.default_up_vector = default_up_vector
        self.default_front_vector = default_front_vector
        self.start = """# .PCD v.7 - Point Cloud Data file format
VERSION .7
FIELDS x y z
SIZE 4 4 4
TYPE F F F
COUNT 1 1 1
WIDTH {0}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {0}
DATA ascii
"""

    def createOBJ(self, input, output, foreground_ratio=0.85, mc_resolution=256, model_save_format='obj', num_points=800000):

        output_dir = output
        os.makedirs(output_dir, exist_ok=True)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        print("Initializing model")
        model = TSR.from_pretrained(
            'stabilityai/TripoSR',
            config_name="config.yaml",
            weight_name="model.ckpt",
        )
        model.to(device)
        print("Model initialized")

        image_path = input
        image = Image.open(image_path)
        try:
            image = resize_foreground(image, foreground_ratio)
        except AssertionError:
            image = image.convert("RGBA")
        image = np.array(image).astype(np.float32) / 255.0
        image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
        image = Image.fromarray((image * 255.0).astype(np.uint8))

        print("Running model")
        with torch.no_grad():
            scene_codes = model([image], device=device)
        print("Model run completed")

        print("Exporting mesh")
        meshes = model.extract_mesh(scene_codes, resolution=mc_resolution)
        meshes[0].export(os.path.join(output_dir, "mesh." + model_save_format))

        mesh = o3d.io.read_triangle_mesh('output/mesh.obj')
        # Define the rotation matrices
        rotation_y = o3d.geometry.get_rotation_matrix_from_axis_angle(
            np.array([0, 1, 0]) * (-np.pi / 2))  # 180 degrees around Y-axis
        rotation_x = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([0, 0, 1]) * (-np.pi / 2))

        # Apply the rotations
        mesh.rotate(rotation_y, center=mesh.get_center())
        mesh.rotate(rotation_x, center=mesh.get_center())
        point_cloud = mesh.sample_points_poisson_disk(number_of_points=num_points)

        # Save the point cloud as a PCD file
        o3d.io.write_point_cloud(f"{output}mesh.pcd", point_cloud)
        # os.remove(output+'mesh.obj')
        print("Mesh exported")

    def extractBoundaryDepth(self, image_path, filename="model.pcd"):
        # kernel = np.array([[-1, -1, -1],
        #                    [-1, 9, -1],
        #                    [-1, -1, -1]])

        input_image = Image.open(image_path)
        depth_map = self.depth_estimator.predictDepth(input_image)
        print("Depth map shape:", depth_map.shape)

        # Convert to PIL Image and display
        img = Image.fromarray(depth_map)
        depth_array = np.array(img)

        # Invert the depth image
        max_depth = np.max(depth_array)
        min_depth = np.min(depth_array)
        inverted_depth_array = max_depth - depth_array + min_depth
        print("Creating the object....")
        self.createObj(inverted_depth_array)
        print("Converting....")
        with open('model.obj', "r") as infile:
            obj = infile.read()
        points = []
        for line in obj.split("\n"):
            if (line != ""):
                line = line.split()
                if (line[0] == "v"):
                    point = [float(line[1]), float(line[2]), float(line[3])]
                    points.append(point)
        with open(filename, "w") as outfile:
            outfile.write(self.start.format(len(points)))

            for point in points:
                outfile.write("{} {} {}\n".format(point[0], point[1], point[2]))



    def verticalPlaneExtraction(self, file):
        """Focus on vertical planes to reduce the 3D boundary extraction problem to 2D."""
        pcd = o3d.io.read_point_cloud(file)
        points = np.asarray(pcd.points)
        return points

    def orthogonalProjection(self, points, vertical_plane_normal=(0, 1, 0), plane_point=(0, 0, 0)):
        """
        Project the 3D points onto a vertical plane defined by vertical_plane_normal and plane_point.
        """
        projected_points_xz = points[:, [0, 2]]

        return projected_points_xz

    def boundaryDelineation(self, points_2d):
        """Precisely delineate the 2D boundary that encapsulates the scene."""
        # Determine the boundary of the points (convex hull)
        hull = ConvexHull(points_2d)
        hull_points = points_2d[hull.vertices]

        # Invert the y-axis to correct upside down issue
        hull_points[:, 1] = -hull_points[:, 1]
        return hull_points

    def polygonApproximation(self, hull_points):
        """Approximate the 2D boundary with a polygon."""
        # Epsilon parameter for approximation accuracy (adjust as needed)
        epsilon = 0.01 * cv2.arcLength(hull_points.astype(np.float32), True)
        approx_polygon = cv2.approxPolyDP(hull_points.astype(np.float32), epsilon, True)
        return approx_polygon

    def visualizeVerticalPlaneExtraction(self, points):
        plt.figure(figsize=(8, 8))
        plt.scatter(points[:, 0], points[:, 1], c='blue', s=1)
        plt.title('Vertical Plane Extraction')
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def visualizeOrthogonicProjection(self, points_2d):
        plt.figure(figsize=(8, 8))
        plt.scatter(points_2d[:, 0], points_2d[:, 1], c='green', s=1)
        plt.title('Orthogonic Projection')
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        # plt.gca().invert_yaxis()
        plt.show()

    def visualizeBoundaryDelineation(self, hull_points):
        plt.figure(figsize=(8, 8))
        plt.plot(hull_points[:, 0], hull_points[:, 1], 'k--', lw=1)
        plt.fill(hull_points[:, 0], hull_points[:, 1], 'lightgray')
        plt.title('Boundary Delineation')
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gca().invert_yaxis()
        plt.show()

    def visualizePolygonApproximation(self, hull_points, approx_polygon):
        plt.figure(figsize=(8, 8))
        plt.plot(hull_points[:, 0], hull_points[:, 1], 'k--', lw=1)
        plt.plot(approx_polygon[:, 0, 0], approx_polygon[:, 0, 1], 'b-', lw=2)
        plt.fill(hull_points[:, 0], hull_points[:, 1], 'lightgray')
        plt.title('Polygon Approximation')
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.gca().invert_yaxis()
        plt.show()

    def visualizePolygonApproximation(self, hull_points, approx_polygon):
        # Connect the hull points with straight lines
        for i in range(len(hull_points)):
            next_index = (i + 1) % len(hull_points)
            plt.plot([hull_points[i, 0], hull_points[next_index, 0]],
                     [hull_points[i, 1], hull_points[next_index, 1]], 'k--', lw=1)

        # Plot the polygon approximation with straight lines
        for i in range(len(approx_polygon)):
            next_index = (i + 1) % len(approx_polygon)
            plt.plot([approx_polygon[i][0, 0], approx_polygon[next_index][0, 0]],
                     [approx_polygon[i][0, 1], approx_polygon[next_index][0, 1]], 'b-', lw=2)

        # Fill the convex hull for visualization
        plt.fill(hull_points[:, 0], hull_points[:, 1], 'lightgray', alpha=0.5)

        # Plot each polygon vertex and annotate with numbers
        for i, vertex in enumerate(approx_polygon[:, 0, :]):
            plt.plot(vertex[0], vertex[1], 'bx')  # Blue 'x' for each vertex
            plt.text(vertex[0], vertex[1], str(i), color='black', fontsize=6, ha='right', va='bottom')

        # Set up the plot
        plt.title('Polygon Approximation of 2D Orthographic Projection')
        plt.xlabel('X axis')
        plt.ylabel('Y axis')
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()

    def create_json_file(self, extrinsic, intrinsic, vertices, file_name='camera_vertices.json'):
        # Extract intrinsic parameters if it's a PinholeCameraIntrinsic object
        if isinstance(intrinsic, o3d.camera.PinholeCameraIntrinsic):
            intrinsic_data = {
                'width': intrinsic.width,
                'height': intrinsic.height,
                'fx': intrinsic.get_focal_length()[0],
                'fy': intrinsic.get_focal_length()[1],
                'cx': intrinsic.get_principal_point()[0],
                'cy': intrinsic.get_principal_point()[1],
                'intrinsic_matrix': intrinsic.intrinsic_matrix.tolist()
            }
        else:
            # Assuming intrinsic is already in a serializable format
            intrinsic_data = intrinsic

        data = {
            'camera': {
                'extrinsic': extrinsic.tolist() if isinstance(extrinsic, np.ndarray) else extrinsic,
                'intrinsic': intrinsic_data
            },
            'vertices': vertices.tolist() if isinstance(vertices, np.ndarray) else vertices
        }

        with open(file_name, 'w') as f:
            json.dump(data, f, indent=4)

    def set_default_view(self, view_control, point_cloud):
        view_control.set_lookat(point_cloud.get_center())
        view_control.set_up(self.default_up_vector)
        view_control.set_front(self.default_front_vector)
        view_control.set_zoom(self.default_zoom)

    def visualizeRenderedScene(self, input, image_path, vertices, renderJson=False):
        # Load the PCD file
        pcd = o3d.io.read_point_cloud(input)

        # Load the image to get its dimensions
        image = Image.open(image_path)
        img_width, img_height = image.size

        # Visualize the point cloud
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=img_width, height=img_height, window_name='Rendered Scene')
        vis.add_geometry(pcd)

        # Get the view control
        view_ctl = vis.get_view_control()

        # Set the default view
        self.set_default_view(view_ctl, pcd)

        # Run the visualizer
        vis.run()
        vis.destroy_window()

        if renderJson:
            cam_params = view_ctl.convert_to_pinhole_camera_parameters()
            self.create_json_file(cam_params.extrinsic, cam_params.intrinsic, vertices)
