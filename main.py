# import json
# from DepthEstimator import BoundaryDepthExtractor
# import open3d as o3d
# import numpy as np
#
#
#
# if __name__ == "__main__":
#     boundaryDepthExtractor = BoundaryDepthExtractor()
#     # image_path = 'images/emptyRoom.png'
#     # boundaryDepthExtractor.createOBJ(image_path, 'output/')
#     points = boundaryDepthExtractor.verticalPlaneExtraction("output/mesh.pcd")
#     points_2d = boundaryDepthExtractor.orthogonalProjection(points)
#     hull_points = boundaryDepthExtractor.boundaryDelineation(points_2d)
#     vertices = boundaryDepthExtractor.polygonApproximation(hull_points)
#     boundaryDepthExtractor.visualizePolygonApproximation(hull_points, vertices)

import open3d as o3d
import numpy as np

# Load the PCD file
pcd = o3d.io.read_point_cloud("output/mesh.pcd")

# Visualize the point cloud
vis = o3d.visualization.Visualizer()
vis.create_window()

# Add the point cloud to the visualizer
vis.add_geometry(pcd)

# Set the view point
view_ctl = vis.get_view_control()
cam_params = view_ctl.convert_to_pinhole_camera_parameters()

# You can adjust these values to change the camera angle and position
cam_params.extrinsic = np.array([[-1,  0,  0, 0],  # Rotate 180 degrees around y-axis
                                 [ 0,  -1,  0, 0],
                                 [ 0,  0, -1, 0.8],  # Invert z-axis to maintain right-handed coordinate system
                                 [ 0,  0,  0, 1]])



view_ctl.convert_from_pinhole_camera_parameters(cam_params)

# Run the visualizer
vis.run()
vis.destroy_window()