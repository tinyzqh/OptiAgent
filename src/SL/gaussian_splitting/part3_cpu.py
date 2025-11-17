import os
import torch
import pycolmap
import numpy as np
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt


from splat.gaussians import Gaussians
from splat.gaussian_scene import GaussianScene
from splat.utils import read_images_text, read_images_binary
from splat.read_colmap import read_images_text, qvec2rotmat
from splat.utils import read_images_text, read_images_binary


colmap_path = "treehill/sparse/0"

reconstruction = pycolmap.Reconstruction(colmap_path)
test = False

points3d = reconstruction.points3D
images = read_images_binary(f"{colmap_path}/images.bin")
cameras = reconstruction.cameras

# we will only use points from image_num 100
image_num = 100

image = images[image_num]
points_in_3d = []
points2d = []
colors2d = []
for idx, point in enumerate(image.xys):
    point3d_id = image.point3D_ids[idx]
    if point3d_id > 0:
        try:
            color = points3d[point3d_id].color
            points_in_3d.append(points3d[point3d_id].xyz)
            points2d.append(point)
            colors2d.append(color)
        except:
            pass

points_in_3d = np.array(points_in_3d)
points2d = np.array(points2d)
colors2d = np.array(colors2d)

gaussians = Gaussians(torch.Tensor(points_in_3d), torch.Tensor(colors2d), model_path="point_clouds")


scene = GaussianScene(colmap_path=colmap_path, gaussians=gaussians)

plt.scatter(points2d[:, 0], points2d[:, 1], c=colors2d / 256, s=1)
plt.xlim(0, 6000)
plt.ylim(0, 3744)
plt.gca().invert_yaxis()
save_path = os.path.join(Path(__file__).resolve().parent, "part3_cpu_1.png")
plt.savefig(save_path)


with torch.no_grad():
    points, colors = scene.render_points_image(image_num)

print(points.shape)
plt.scatter(points[:, 0].cpu(), points[:, 1].cpu(), c=colors.cpu(), s=1)
plt.xlim(0, 6000)
plt.ylim(0, 3744)
plt.gca().invert_yaxis()
save_path = os.path.join(Path(__file__).resolve().parent, "part3_cpu_2.png")
plt.savefig(save_path)


image = scene.render_image(image_num, tile_size=2)

import matplotlib.pyplot as plt

# Assuming new_image is your image tensor

# Set background to white
new_image = image * (image > 0.001).float()
new_image[new_image <= 0.001] = 1

# Display the image
plt.imshow(new_image.detach().transpose(0, 1), cmap="gray", vmin=0, vmax=1)
save_path = os.path.join(Path(__file__).resolve().parent, "part3_cpu_3.png")
plt.savefig(save_path)


# shows there are points in the white space that we just cannot see
image[:1000].max()


colmap_path = "treehill/sparse/0"
reconstruction = pycolmap.Reconstruction(colmap_path)

points3d = reconstruction.points3D

all_points3d = []
all_point_colors = []

for idx, point in enumerate(points3d.values()):
    if point.track.length() >= 2:
        all_points3d.append(point.xyz)
        all_point_colors.append(point.color)

gaussians = Gaussians(torch.Tensor(all_points3d), torch.Tensor(all_point_colors), model_path="point_clouds")
scene = GaussianScene(colmap_path=colmap_path, gaussians=gaussians)


image_num = 100

processed_scene = scene.preprocess(image_num)
plt.scatter(processed_scene.points_xy[:, 0].detach().cpu(), processed_scene.points_xy[:, 1].detach().cpu(), c=processed_scene.colors.detach().cpu(), s=1)
plt.xlim(0, 6000)
plt.ylim(0, 3744)
plt.gca().invert_yaxis()
save_path = os.path.join(Path(__file__).resolve().parent, "part3_cpu_4.png")
plt.savefig(save_path)


# this may take a while
with torch.no_grad():
    image = scene.render_image(image_num)

plt.imshow(image.cpu().detach().transpose(0, 1) * 256)
save_path = os.path.join(Path(__file__).resolve().parent, "part3_cpu_5.png")
plt.savefig(save_path)
