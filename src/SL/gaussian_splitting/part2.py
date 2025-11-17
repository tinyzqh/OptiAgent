import math
import torch
import pycolmap
import numpy as np
import torch.nn as nn


from splat.utils import ndc2Pix
from splat.gaussians import Gaussians
from splat.gaussian_scene import GaussianScene
from splat.utils import read_images_text, read_images_binary
from splat.utils import read_camera_file, read_image_file, build_rotation, in_view_frustum


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

colmap_path = "treehill/sparse/0"
reconstruction = pycolmap.Reconstruction(colmap_path)

points3d = reconstruction.points3D
images = read_images_binary(f"{colmap_path}/images.bin")
cameras = reconstruction.cameras

all_points3d = []
all_point_colors = []

for idx, point in enumerate(points3d.values()):
    if point.track.length() >= 2:
        all_points3d.append(point.xyz)
        all_point_colors.append(point.color)

gaussians = Gaussians(torch.Tensor(all_points3d), torch.Tensor(all_point_colors), model_path="point_clouds")


def get_extrinsic_matrix(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    Get the homogenous extrinsic matrix for the camera
    """
    Rt = torch.zeros((4, 4))
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt


def getIntinsicMatrix(
    focal_x: torch.Tensor, focal_y: torch.Tensor, height: torch.Tensor, width: torch.Tensor, zfar: torch.Tensor = torch.Tensor([100.0]), znear: torch.Tensor = torch.Tensor([0.001])
) -> torch.Tensor:
    """
    Gets the internal perspective projection matrix

    znear: near plane set by user
    zfar: far plane set by user
    fovX: field of view in x, calculated from the focal length
    fovY: field of view in y, calculated from the focal length
    """
    fovX = torch.Tensor([2 * math.atan(width / (2 * focal_x))])
    fovY = torch.Tensor([2 * math.atan(height / (2 * focal_y))])

    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def compute_2d_covariance(
    points: torch.Tensor, extrinsic_matrix: torch.Tensor, covariance_3d: torch.Tensor, tan_fovY: torch.Tensor, tan_fovX: torch.Tensor, focal_x: torch.Tensor, focal_y: torch.Tensor
) -> torch.Tensor:
    """
    Compute the 2D covariance matrix for each gaussian
    """
    points = torch.cat([points, torch.ones(points.shape[0], 1, device=points.device)], dim=1)
    points_transformed = (points @ extrinsic_matrix)[:, :3]
    limx = 1.3 * tan_fovX
    limy = 1.3 * tan_fovY
    x = points_transformed[:, 0] / points_transformed[:, 2]
    y = points_transformed[:, 1] / points_transformed[:, 2]
    z = points_transformed[:, 2]
    x = torch.clamp(x, -limx, limx) * z
    y = torch.clamp(y, -limy, limy) * z

    J = torch.zeros((points_transformed.shape[0], 3, 3), device=covariance_3d.device)
    J[:, 0, 0] = focal_x / z
    J[:, 0, 2] = -(focal_x * x) / (z ** 2)
    J[:, 1, 1] = focal_y / z
    J[:, 1, 2] = -(focal_y * y) / (z ** 2)

    # transpose as originally set up for perspective projection
    # so we now transform back
    W = extrinsic_matrix[:3, :3].T

    return (J @ W @ covariance_3d @ W.T @ J.transpose(1, 2))[:, :2, :2]


covariance_3d = gaussians.get_3d_covariance_matrix()

# we will examine the 100th image
image_num = 100
image_dict = read_image_file(colmap_path)
camera_dict = read_camera_file(colmap_path)

# convert quaternion to rotation matrix
rotation_matrix = build_rotation(torch.Tensor(image_dict[image_num].qvec).unsqueeze(0))
translation = torch.Tensor(image_dict[image_num].tvec).unsqueeze(0)
extrinsic_matrix = get_extrinsic_matrix(rotation_matrix, translation).to(device)
focal_x, focal_y = camera_dict[image_dict[image_num].camera_id].params[:2]
c_x, c_y = camera_dict[image_dict[image_num].camera_id].params[2:4]
width = camera_dict[image_dict[image_num].camera_id].width
height = camera_dict[image_dict[image_num].camera_id].height

# note we transpose the intrinsic matrix
intrinsic_matrix = getIntinsicMatrix(focal_x, focal_y, height, width).T.to(device)
in_view = in_view_frustum(points=gaussians.points, view_matrix=extrinsic_matrix.T)

fovX = torch.Tensor([2 * math.atan(width / (2 * focal_x))]).to(device)
fovY = torch.Tensor([2 * math.atan(height / (2 * focal_y))]).to(device)

covariance_2d = compute_2d_covariance(
    points=gaussians.points[in_view],
    extrinsic_matrix=extrinsic_matrix.T,
    covariance_3d=covariance_3d[in_view],
    tan_fovY=torch.tan(fovX / 2),
    tan_fovX=torch.tan(fovX / 2),
    focal_x=focal_x,
    focal_y=focal_y,
)


def compute_inverted_covariance(covariance_2d: torch.Tensor) -> torch.Tensor:
    """
    Compute the inverse covariance matrix

    For a 2x2 matrix
    given as
    [[a, b],
     [c, d]]
     the determinant is ad - bc

    To get the inverse matrix reshuffle the terms like so
    and multiply by 1/determinant
    [[d, -b],
     [-c, a]] * (1 / determinant)
    """
    determinant = covariance_2d[:, 0, 0] * covariance_2d[:, 1, 1] - covariance_2d[:, 0, 1] * covariance_2d[:, 1, 0]
    determinant = torch.clamp(determinant, min=1e-3)
    inverse_covariance = torch.zeros_like(covariance_2d)
    inverse_covariance[:, 0, 0] = covariance_2d[:, 1, 1] / determinant
    inverse_covariance[:, 1, 1] = covariance_2d[:, 0, 0] / determinant
    inverse_covariance[:, 0, 1] = -covariance_2d[:, 0, 1] / determinant
    inverse_covariance[:, 1, 0] = -covariance_2d[:, 1, 0] / determinant
    return inverse_covariance


def compute_extent_and_radius(covariance_2d: torch.Tensor):
    mid = 0.5 * (covariance_2d[:, 0, 0] + covariance_2d[:, 1, 1])
    det = covariance_2d[:, 0, 0] * covariance_2d[:, 1, 1] - covariance_2d[:, 0, 1] ** 2
    intermediate_matrix = (mid * mid - det).view(-1, 1)
    intermediate_matrix = torch.cat([intermediate_matrix, torch.ones_like(intermediate_matrix) * 0.1], dim=1)

    max_values = torch.max(intermediate_matrix, dim=1).values
    lambda1 = mid + torch.sqrt(max_values)
    lambda2 = mid - torch.sqrt(max_values)
    # now we have the eigenvalues, we can calculate the max radius
    max_radius = torch.ceil(3.0 * torch.sqrt(torch.max(lambda1, lambda2)))

    return max_radius


inverted_covariance = compute_inverted_covariance(covariance_2d)
extent = compute_extent_and_radius(covariance_2d)


splat = gaussians.points[in_view][:1]
splat_2d_covariance = covariance_2d[:1]

# project the splat to 2D
homogeneous_splat = torch.cat([splat, torch.ones(splat.shape[0], 1, device=splat.device)], dim=1)
temp_splat = homogeneous_splat @ extrinsic_matrix.T @ intrinsic_matrix
splat_image_plane = temp_splat[:, :3] / temp_splat[:, 3].unsqueeze(1)
splat_xy = splat_image_plane[:, :2]

# convert to pixel coordinates from normalized device coordinates
splat_xy[:, 0] = ndc2Pix(splat_xy[:, 0], width)
splat_xy[:, 1] = ndc2Pix(splat_xy[:, 1], height)
inverted_splat_2d_covariance = compute_inverted_covariance(splat_2d_covariance)
radius = compute_extent_and_radius(splat_2d_covariance)


def compute_gaussian_weight(pixel_coord: torch.Tensor, point_mean: torch.Tensor, inverse_covariance: torch.Tensor) -> torch.Tensor:  # (1, 2) tensor

    difference = point_mean - pixel_coord
    power = -0.5 * difference @ inverse_covariance @ difference.T
    return torch.exp(power).item()


strength = compute_gaussian_weight(pixel_coord=splat_xy - 0.01, point_mean=splat_xy, inverse_covariance=inverted_splat_2d_covariance)
print(f"The strength with a .1 pixel offset in the x direction is {strength}")
strength = compute_gaussian_weight(pixel_coord=splat_xy - 1, point_mean=splat_xy, inverse_covariance=inverted_splat_2d_covariance)
print(f"The strength with a 1 pixel offset in the x direction is {strength}")
strength = compute_gaussian_weight(pixel_coord=splat_xy - 10, point_mean=splat_xy, inverse_covariance=inverted_splat_2d_covariance)
print(f"The strength with a 10 pixel offset in the x direction is {strength}")
