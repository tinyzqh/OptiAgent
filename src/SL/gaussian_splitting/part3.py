import os
from pathlib import Path

from splat.read_colmap import read_images_text, qvec2rotmat
from splat.gaussians import Gaussians
from splat.gaussian_scene import GaussianScene
import pycolmap
import torch
import torch.nn as nn
import numpy as np

from splat.utils import read_images_text, read_images_binary

import os
from matplotlib import pyplot as plt


colmap_path = "treehill/sparse/0"

reconstruction = pycolmap.Reconstruction(colmap_path)
test = False

points3d = reconstruction.points3D
images = read_images_binary(f"{colmap_path}/images.bin")
cameras = reconstruction.cameras

all_points3d = []
all_point_colors = []
image_num = 100

for idx, point in enumerate(points3d.values()):
    # I would assume this is seeing it from multiple images
    if point.track.length() >= 2:
        all_points3d.append(point.xyz)
        all_point_colors.append(point.color)


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

all_points3d = np.array(all_points3d)
all_point_colors = np.array(all_point_colors)
points_in_3d = np.array(points_in_3d)
points2d = np.array(points2d)
colors2d = np.array(colors2d)

print(all_points3d.shape, points_in_3d.shape)
gaussians = Gaussians(torch.Tensor(all_points3d), torch.Tensor(all_point_colors), model_path="point_clouds")

import matplotlib.pyplot as plt

scene = GaussianScene(colmap_path=colmap_path, gaussians=gaussians)

plt.scatter(points2d[:, 0], points2d[:, 1], c=colors2d / 256, s=1)
plt.xlim(0, 6000)
plt.ylim(0, 3744)
plt.gca().invert_yaxis()
save_path = os.path.join(Path(__file__).resolve().parent, "part3_1.png")
plt.savefig(save_path)


import os
from pathlib import Path
from torch.utils.cpp_extension import load_inline

project_path = Path(__file__).resolve().parent
cuda_src = Path(os.path.join(project_path, "splat/c/render.cu")).read_text()

cpp_src = """
torch::Tensor render_image(
    int image_height,
    int image_width,
    int tile_size,
    torch::Tensor point_means,
    torch::Tensor point_colors,
    torch::Tensor inverse_covariance_2d,
    torch::Tensor min_x,
    torch::Tensor max_x,
    torch::Tensor min_y,
    torch::Tensor max_y,
    torch::Tensor opacity);
"""

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def load_cuda(cuda_src, cpp_src, funcs, opt=True, verbose=False):
    return load_inline(name="inline_ext", cpp_sources=[cpp_src], cuda_sources=[cuda_src], functions=funcs, extra_cuda_cflags=["-O1"] if opt else [], verbose=verbose)


# Set MAX_JOBS for parallel compilation
os.environ["MAX_JOBS"] = "10"

# Compile and load the CUDA module
module = load_cuda(cuda_src, cpp_src, ["render_image"], opt=True, verbose=True)


image = scene.render_image_cuda(image_num, tile_size=16)


plt.imshow(image.detach().cpu() * 256)

save_path = os.path.join(Path(__file__).resolve().parent, "part3_2.png")
plt.savefig(save_path)
