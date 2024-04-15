# Depth Estimation and Boundary Extraction

This project implements a multi-step pipeline for extracting boundary depth from images. The process involves estimating a depth map from a monocular image, back-projecting this image into a 3D triangular mesh, and then extracting planar surfaces to approximate boundary conditions. This approach is inspired by the methodology described in the paper "LOOSECONTROL: Lifting ControlNet for Generalized Depth Conditioning."

## Installation

Before running the script, ensure that you have python env in the same directory and it's activated:

```bash
pip install -r requirements.txt
```

## How It Works:

1. **Depth Map Estimation:** The first step is to estimate the depth map of the given image using a monocular depth estimator. In the provided code, this is accomplished using the ```DepthEstimator``` class, which utilizes the ```DPTForDepthEstimation``` model from the transformers library. The input image is processed and passed through the model to obtain a depth map.

2. **3D Triangular Mesh Back-Projection:** Once the depth map is obtained, the next step is to back-project the image into a 3D triangular mesh within the world space. This involves converting the depth map into a set of 3D points that represent the scene geometry. In the provided code, this step is performed by the createObj method within the ```BoundaryDepthExtractor``` class, which generates a 3D object file (model.obj) from the depth map.
   
3. **Vertical Plane Extraction:** For efficiency during training, the code focuses only on vertical planes. This reduces the 3D boundary extraction problem to a simpler 2D problem. The ```verticalPlaneExtraction``` method in the ```BoundaryDepthExtractor``` class is responsible for this step, although the actual implementation is not provided in the code snippet.
   
4. **Orthographic Projection:** The 3D mesh of the scene is projected onto a horizontal plane using orthographic projection. This projection facilitates the precise delineation of the 2D boundary that encapsulates the scene. The ```orthogonalProjection``` method in the ```BoundaryDepthExtractor``` class performs this step by projecting the 3D points onto a vertical plane.
   
5. **2D Boundary Delineation:** After projection, the next step is to delineate the 2D boundary that encapsulates the scene. This is achieved by determining the convex hull of the projected points, which represents the outer boundary of the scene. The ```boundaryDelineation``` method in the ```BoundaryDepthExtractor``` class performs this step.
   
6. **Polygon Approximation:** The 2D boundary is then approximated with a polygon to simplify the representation. This approximation is done using the Douglas-Peucker algorithm, which reduces the number of points in the boundary while maintaining its overall shape. The ```polygonApproximation``` method in the ```BoundaryDepthExtractor``` class performs this step.

## Input (1242 x 822):
<img width="500" alt="emptyRoom" src="https://github.com/ritessshhh/BoundaryDepthExtraction/assets/81812754/fcfa2a85-f5b6-41e6-96c8-8734bbf6db98">

## Polygon Approximation:
<img width="500" alt="Screenshot 2024-04-05 at 2 14 29 AM" src="https://github.com/ritessshhh/BoundaryDepthExtraction/assets/81812754/6aa05bf7-7d51-41e6-b4bc-eee72bb48528">

## Rerendered Image (1242 x 822)
<img width="500" alt="Screenshot 2024-04-15 at 1 12 38 AM" src="https://github.com/ritessshhh/BoundaryDepthExtraction/assets/81812754/8f8fefd7-2cbf-4259-8b59-a17a72abceb6">

## Camera Parameters and Vertices(Saved as a JSON file):
```bash
{
    "camera": {
        "extrinsic": [
            [
                1.0,
                0.0,
                0.0,
                0.03777360018751525
            ],
            [
                -0.0,
                -1.0,
                -0.0,
                0.0074896122114004515
            ],
            [
                -0.0,
                -0.0,
                -1.0,
                0.9023693128733252
            ],
            [
                0.0,
                0.0,
                0.0,
                1.0
            ]
        ],
        "intrinsic": {
            "width": 1242,
            "height": 822,
            "fx": 711.8728819108087,
            "fy": 711.8728819108087,
            "cx": 620.5,
            "cy": 410.5,
            "intrinsic_matrix": [
                [
                    711.8728819108087,
                    0.0,
                    620.5
                ],
                [
                    0.0,
                    711.8728819108087,
                    410.5
                ],
                [
                    0.0,
                    0.0,
                    1.0
                ]
            ]
        }
    },
    "vertices": [
        [
            [
                0.461700439453125,
                -0.45669275522232056
            ]
        ],
        [
            [
                -0.5396929979324341,
                -0.4580613672733307
            ]
        ],
        [
            [
                -0.5212900042533875,
                0.35700321197509766
            ]
        ],
        [
            [
                0.43428346514701843,
                0.36788055300712585
            ]
        ]
    ]
}
```


