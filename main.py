from DepthEstimator import BoundaryDepthExtractor


if __name__ == "__main__":
    boundaryDepthExtractor = BoundaryDepthExtractor()
    image_path = 'images/emptyRoom.png'
    boundaryDepthExtractor.createOBJ(image_path, 'output/')
    points = boundaryDepthExtractor.verticalPlaneExtraction("output/mesh.pcd")
    points_2d = boundaryDepthExtractor.orthogonalProjection(points)
    hull_points = boundaryDepthExtractor.boundaryDelineation(points_2d)
    vertices = boundaryDepthExtractor.polygonApproximation(hull_points)
    boundaryDepthExtractor.visualizePolygonApproximation(hull_points, vertices)
    boundaryDepthExtractor.visualizeRenderedScene('output/mesh.pcd', vertices, renderJson=True)