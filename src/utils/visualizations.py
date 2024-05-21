
import os  # nopep8
os.environ["PYOPENGL_PLATFORM"] = "egl"  # nopep8

import mcubes as mc
from src.utils.util import to_point_list
import trimesh.scene
from matplotlib import cm, colors
import trimesh
from pathlib import Path
from torch.nn import Sigmoid
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import k3d
import io
from IPython.display import Image as ImageDisplay
import torchvision.utils as vutils
from termcolor import cprint


def tensor2im(image_tensor, imtype=np.uint8):
    # image_numpy = image_tensor[0].cpu().float().numpy()
    # if image_numpy.shape[0] == 1:
    #     image_numpy = np.tile(image_numpy, (3, 1, 1))
    # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    # return image_numpy.astype(imtype)

    n_img = min(image_tensor.shape[0], 16)
    image_tensor = image_tensor[:n_img]

    if image_tensor.shape[1] == 1:
        image_tensor = image_tensor.repeat(1, 3, 1, 1)

    # if image_tensor.shape[1] == 4:
        # import pdb; pdb.set_trace()

    image_tensor = vutils.make_grid(image_tensor, nrow=4)

    image_numpy = image_tensor.cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)
    cprint(f"{image_path} saved", "blue")


def visualize_occupancy(occupancy_grid, flip_axes=False):
    point_list = np.concatenate([c[:, np.newaxis]
                                 for c in np.where(occupancy_grid)], axis=1)

    visualize_pointcloud(
        point_list, 1, flip_axes=flip_axes, name='occupancy_grid')


def visualize_pointcloud(point_cloud, point_size, colors=None, flip_axes=False, name='point_cloud'):
    plot = k3d.plot(name=name, grid_visible=False,
                    grid=(-0.55, -0.55, -0.55, 0.55, 0.55, 0.55))
    if flip_axes:
        point_cloud[:, 2] = point_cloud[:, 2] * -1
        point_cloud[:, [0, 1, 2]] = point_cloud[:, [0, 2, 1]]
    plt_points = k3d.points(positions=point_cloud.astype(
        np.float32), point_size=point_size, colors=colors if colors is not None else [], color=0xd0d0d0)
    plot += plt_points
    plt_points.shader = '3d'
    plot.display()
    plt.show()


def visualize_image(image_array):
    plt.imshow(image_array)


sigm = Sigmoid()


def save_voxels(pred, gt, save_path, iteration, is_train=True):
    pred_plots = plot_voxels(sigm(pred.detach()), rot02=1, rot12=1)

    gt_plots = plot_voxels(gt.detach(), rot02=1, rot12=1)
    title = "train"
    if (not is_train):
        title = "validation"
    fig = visualize_png(gt_plots + pred_plots,
                        f"{title}/Target-Reconstruction", rows=2)
    final_save_path = f"{save_path}/{title}_{int(iteration)}"
    print(final_save_path, "saved")
    fig.savefig(final_save_path)
    print(final_save_path, "saved")
    return fig


def plot_voxels(voxels_input, rot01=0, rot02=0, rot12=0, nimgs=3):
    output = []
    for i in range(nimgs):
        voxels = voxels_input[i]
        # import pdb; pdb.set_trace()
        voxels[voxels >= 0.5] = 1
        voxels[voxels < 0.5] = 0
        voxels = voxels.rot90(rot01, (0, 1))
        voxels = voxels.rot90(rot02, (0, 2))
        voxels = voxels.rot90(rot12, (1, 2))
        ax = plt.figure(figsize=(20, 20)).add_subplot(projection='3d')
        ax.set_box_aspect((1, 1, 1))
        ax.voxels(voxels)
        buf = io.BytesIO()
        ax.grid(False)
        ax.axison = False
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.savefig(buf)
        buf.seek(0)
        img = Image.open(buf)
        plt.clf()
        plt.close()
        output.append(img)

    return output


def visualize_images(images, rows=5):
    nimages = images.shape[0]
    columns = nimages // rows
    if (nimages % rows != 0):
        rows += 1
    if (columns == 0):
        columns = 1
    fig = plt.figure(figsize=(20, 20))
    # fig.suptitle('ss', fontsize=20)
    for i, img in enumerate(images):
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(img)
    # plt.show()
    return fig


def visualize_png(images, title, rows=3):
    nimages = len(images)
    columns = nimages // rows
    if (nimages % rows != 0):
        rows += 1
    if (columns == 0):
        columns = 1

    fig = plt.figure(figsize=(10, 4.8))
    fig.suptitle(title, fontsize=20)
    for i, img in enumerate(images):
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(img)

    plt.show()
    return fig


def visualize_png2(images, title, rows=5):
    nimages = len(images)
    columns = nimages // rows
    if (nimages % rows != 0):
        rows += 1
    if (columns == 0):
        columns = 1

    fig = plt.figure(figsize=(10, 4.8))
    fig.suptitle(title, fontsize=20)
    for i, img in enumerate(images):
        fig.add_subplot(rows, columns, i + 1)
        plt.imshow(img)

    for ax in fig.axes:
        ax.axison = False
    plt.show()
    return fig


def visualize_sdf(sdf: np.array, filename: Path) -> None:
    assert sdf.shape[0] == sdf.shape[1] == sdf.shape[2], "SDF grid has to be of cubic shape"
    print(f"Creating SDF visualization for {sdf.shape[0]}^3 grid ...")

    voxels = np.stack(np.meshgrid(range(sdf.shape[0]), range(
        sdf.shape[1]), range(sdf.shape[2]))).reshape(3, -1).T

    sdf[sdf < 0] /= np.abs(sdf[sdf < 0]).max() if np.sum(sdf < 0) > 0 else 1.
    sdf[sdf > 0] /= sdf[sdf > 0].max() if np.sum(sdf < 0) > 0 else 1.
    sdf /= -2.

    corners = np.array([
        [-.25, -.25, -.25],
        [.25, -.25, -.25],
        [-.25, .25, -.25],
        [.25, .25, -.25],
        [-.25, -.25, .25],
        [.25, -.25, .25],
        [-.25, .25, .25],
        [.25, .25, .25]
    ])[np.newaxis, :].repeat(voxels.shape[0], axis=0).reshape(-1, 3)

    scale_factors = sdf[tuple(voxels.T)].repeat(8, axis=0)
    cube_vertices = voxels.repeat(
        8, axis=0) + corners * scale_factors[:, np.newaxis]
    cube_vertex_colors = cm.get_cmap('seismic')(
        colors.Normalize(vmin=-1, vmax=1)(scale_factors))[:, :3]

    faces = np.array([
        [1, 0, 2], [2, 3, 1], [5, 1, 3], [3, 7, 5], [4, 5, 7], [7, 6, 4],
        [0, 4, 6], [6, 2, 0], [3, 2, 6], [6, 7, 3], [5, 4, 0], [0, 1, 5]
    ])[np.newaxis, :].repeat(voxels.shape[0], axis=0).reshape(-1, 3)
    cube_faces = faces + (np.arange(0, voxels.shape[0]) * 8)[
        np.newaxis, :].repeat(12, axis=0).T.flatten()[:, np.newaxis]

    mesh = trimesh.Trimesh(vertices=cube_vertices, faces=cube_faces,
                           vertex_colors=cube_vertex_colors, process=False)
    img = visualize_mesh(cube_vertices, cube_faces, flip_axes=True)
    mesh.export(str(filename))
    print(f"Exported to {filename}")


def visualize_mesh(vertices, faces, flip_axes=False):
    plot = k3d.plot(name='points', grid_visible=False,
                    grid=(-0.55, -0.55, -0.55, 0.55, 0.55, 0.55))

    # vertices[:, 2] = vertices[:, 2] * -1
    # vertices[:, [0, 1, 2]] = vertices[:, [0, 2, 1]]
    if flip_axes:
        # vertices[:, 2] = vertices[:, 2] * -1
        vertices[:, [0, 1, 2]] = vertices[:, [2, 0, 1]]
       # vertices[:, 1] = vertices[:, 1] * -1
        # vertices[:, [0, 1, 2]] = vertices[:, [1, 0, 2]]

    plt_mesh = k3d.mesh(vertices.astype(np.float32),
                        faces.astype(np.uint32), color=0xd0d0d0)

    plot += plt_mesh
    plt_mesh.shader = '3d'

    plot.display()


def visualize_mesh_file(filePath, flip_axes=False):
    mesh = trimesh.load_mesh(filePath)
    vertices = mesh.vertices
    faces = mesh.faces
    visualize_mesh(vertices, faces, flip_axes=flip_axes)
    print(vertices.shape)
    print(mesh)


def visualize_sdf_as_voxels(sdf, output_path, level=0.5):
    point_list = to_point_list(sdf <= level)
    if point_list.shape[0] > 0:
        base_mesh = trimesh.voxel.ops.multibox(centers=point_list, pitch=1)
        base_mesh.export(output_path)


def visualize_sdf_as_mesh(sdf, output_path, level=0.75, scale_factor=1, saveMesh=False):
    vertices, triangles = mc.marching_cubes(sdf, level)
    vertices = vertices / scale_factor
    if (saveMesh):
        mc.export_obj(vertices, triangles, output_path)

    visualize_mesh(vertices, triangles, flip_axes=True)
