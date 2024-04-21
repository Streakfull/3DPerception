from pathlib import Path
import io
import numpy as np
import k3d
from matplotlib import cm, colors
import PIL.Image
import IPython.display
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.nn import Sigmoid


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
    #import pdb;pdb.set_trace();
    plot.display()
    plt.show()


def visualize_image(image_array):
    plt.imshow(image_array)
    
sigm = Sigmoid()
def save_voxels(pred, gt, save_path, iteration, is_train = True):
     pred_plots = plot_voxels(sigm(pred.detach()), rot02=1,rot12=1)

     gt_plots = plot_voxels(gt.detach(), rot02=1,rot12=1)
     title = "train"
     if(not is_train):
        title = "validation"
     fig = visualize_png(gt_plots + pred_plots, f"{title}/Target-Reconstruction", rows=2)
     final_save_path = f"{save_path}/{title}_{int(iteration)}"
     print(final_save_path, "saved")
     fig.savefig(final_save_path)
     print(final_save_path, "saved")
     return fig
    
def plot_voxels(voxels_input, rot01=0, rot02=0, rot12=0, nimgs=3):
    output = []
    for i in range(nimgs):
        voxels = voxels_input[i]
    #import pdb; pdb.set_trace()
        voxels[voxels >= 0.5] = 1
        voxels[voxels < 0.5] = 0
        voxels = voxels.rot90(rot01, (0, 1))
        voxels = voxels.rot90(rot02, (0, 2))
        voxels = voxels.rot90(rot12, (1, 2))
        ax = plt.figure(figsize=(20,20)).add_subplot(projection='3d')
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
    columns = nimages//rows
    if (nimages % rows != 0):
        rows += 1
    if(columns == 0):
        columns = 1
    fig = plt.figure(figsize=(20, 20))
    #fig.suptitle('ss', fontsize=20)
    for i, img in enumerate(images):
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(img)
    #plt.show()
    return fig

def visualize_png(images, title, rows=3):
    nimages = len(images)
    columns = nimages//rows
    if (nimages % rows != 0):
        rows += 1
    if(columns == 0):
        columns = 1
   
    fig = plt.figure(figsize=(10,4.8))
    fig.suptitle(title, fontsize=20)
    for i, img in enumerate(images):
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(img)
    
#     for ax in fig.axes:
#         ax.axison = False
    #plt.show()
    plt.show()
    return fig

def visualize_png2(images, title, rows=5):
    nimages = len(images)
    columns = nimages//rows
    if (nimages % rows != 0):
        rows += 1
    if(columns == 0):
        columns = 1
   
    fig = plt.figure(figsize=(10,4.8))
    fig.suptitle(title, fontsize=20)
    for i, img in enumerate(images):
        fig.add_subplot(rows, columns, i+1)
        plt.imshow(img)
    
    for ax in fig.axes:
        ax.axison = False
    #plt.show()
    plt.show()
    return fig

