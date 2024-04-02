import json
from DepthEstimator import BoundaryDepthExtractor
import open3d as o3d
import numpy as np



if __name__ == "__main__":
    boundaryDepthExtractor = BoundaryDepthExtractor()

    image_path = 'images/emptyRoom.png'
    boundaryDepthExtractor.createOBJ(image_path, 'output/')

    o3d.visualization.draw_geometries([o3d.io.read_point_cloud('output/output.pcd')])
    # points = boundaryDepthExtractor.verticalPlaneExtraction("model.pcd")
    #
    # # boundaryDepthExtractor.visualizeVerticalPlaneExtraction(points)
    #
    # points_2d = boundaryDepthExtractor.orthogonalProjection(points)
    # # boundaryDepthExtractor.visualizeOrthogonicProjection(points_2d)
    #
    # hull_points = boundaryDepthExtractor.boundaryDelineation(points_2d)
    # # boundaryDepthExtractor.visualizeBoundaryDelineation(hull_points)
    #
    # vertices = boundaryDepthExtractor.polygonApproximation(hull_points)
    # boundaryDepthExtractor.visualizePolygonApproximation(hull_points, vertices)