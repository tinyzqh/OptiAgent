import os
import torch
import pycolmap
import numpy as np
import torch.nn as nn
from pathlib import Path
from matplotlib import pyplot as plt

from splat.gaussians import Gaussians
from splat.gaussian_scene import GaussianScene
from splat.utils import read_images_text, read_images_binary
from splat.utils import read_camera_file, read_image_file, build_rotation

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

os.makedirs("point_clouds", exist_ok=True)
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


def get_intrinsic_matrix(f_x: float, f_y: float, c_x: float, c_y: float) -> torch.Tensor:
    """
    Get the homogenous intrinsic matrix for the camera
    """
    return torch.Tensor([[f_x, 0, c_x, 0], [0, f_y, c_y, 0], [0, 0, 1, 0]])


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
intrinsic_matrix = get_intrinsic_matrix(focal_x, focal_y, c_x, c_y).to(device)


def project_points(points: torch.Tensor, intrinsic_matrix: torch.Tensor, extrinsic_matrix: torch.Tensor) -> torch.Tensor:
    """
    Project the points to the image plane

    Args:
        points: Nx3 tensor
        intrinsic_matrix: 3x4 tensor
        extrinsic_matrix: 4x4 tensor
    """
    homogeneous = torch.ones((4, points.shape[0]), device=points.device)
    homogeneous[:3, :] = points.T
    projected_to_camera_perspective = extrinsic_matrix @ homogeneous
    projected_to_image_plane = (intrinsic_matrix @ projected_to_camera_perspective).T  # Nx4

    x = projected_to_image_plane[:, 0] / projected_to_image_plane[:, 2]
    y = projected_to_image_plane[:, 1] / projected_to_image_plane[:, 2]
    return x, y


points = project_points(gaussians.points, intrinsic_matrix, extrinsic_matrix)
plt.scatter(points[0].cpu().detach(), points[1].cpu().detach(), c=gaussians.colors.cpu().detach(), s=1)
plt.xlim(0, camera_dict[image_dict[image_num].camera_id].width)
plt.ylim(0, camera_dict[image_dict[image_num].camera_id].height)
plt.gca().invert_yaxis()
save_path = os.path.join(Path(__file__).resolve().parent, "part1.png")
plt.savefig(save_path)
