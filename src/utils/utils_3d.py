from pytorch3d.structures import Meshes
import pytorch3d
from pytorch3d.structures import Pointclouds, Meshes
from termcolor import cprint
import mcubes as mc
import numpy as np
import trimesh
import torchvision.utils as vutils
import imageio
import einops
from einops import rearrange, repeat

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
)

from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
)

from pytorch3d.transforms import RotateAxisAngle
from pytorch3d.structures import Meshes
import torch


def init_mesh_renderer(image_size=512, dist=3.5, elev=90, azim=90, camera='0', device='cuda:0'):
    # Initialize a camera.
    # With world coordinates +Y up, +X left and +Z in, the front of the cow is facing the -Z direction.
    # So we move the camera by 180 in the azimuth direction so it is facing the front of the cow.

    if camera == '0':
        # for vox orientation
        # dist, elev, azim = 1.7, 20, 20 # shapenet
        # dist, elev, azim = 3.5, 90, 90 # front view

        # dist, elev, azim = 3.5, 0, 135 # front view
        camera_cls = FoVPerspectiveCameras
    else:
        # dist, elev, azim = 5, 45, 135 # shapenet
        camera_cls = FoVOrthographicCameras

    R, T = look_at_view_transform(dist, elev, azim)
    cameras = camera_cls(device=device, R=R, T=T)

    # Define the settings for rasterization and shading. Here we set the output image to be of size
    # 512x512. As we are rendering images for visualization purposes only we will set faces_per_pixel=1
    # and blur_radius=0.0. We also set bin_size and max_faces_per_bin to None which ensure that
    # the faster coarse-to-fine rasterization method is used. Refer to rasterize_meshes.py for
    # explanations of these parameters. Refer to docs/notes/renderer.md for an explanation of
    # the difference between naive and coarse-to-fine rasterization.
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0,
        faces_per_pixel=1,
    )

    # Place a point light in front of the object. As mentioned above, the front of the cow is facing the
    # -z direction.
    lights = PointLights(device=device, location=[[1.0, 1.0, 0.0]])

    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and
    # apply the Phong lighting model
    # renderer = MeshRenderer(
    #     rasterizer=MeshRasterizer(
    #         cameras=cameras,
    #         raster_settings=raster_settings
    #     ),
    #     shader=SoftPhongShader(
    #         device=device,
    #         cameras=cameras,
    #         lights=lights
    #     )
    # )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, raster_settings=raster_settings),
        shader=HardPhongShader(device=device, cameras=cameras)
    )
    return renderer


def render_pcd(renderer, verts, color=[1, 1, 1], alpha=False):
    if verts.dim() == 2:
        verts = verts[None, ...]

    verts = verts.to(renderer.rasterizer.cameras.device)
    # verts = verts.cpu()

    # verts: tensor of shape: B, V, 3
    # return: image tensor with shape: B, C, H, W
    V = verts.shape[1]
    B = verts.shape[0]
    features = torch.ones_like(verts)
    for i in range(3):
        features[:, :, i] = color[i]
    pcl = Pointclouds(points=verts, features=features)
    try:
        images = renderer(pcl)
    except:
        images = renderer(pcl, gamma=(1e-4,),)

    return images.permute(0, 3, 1, 2)


def render_mesh(renderer, mesh, color=None, norm=True):
    # verts: tensor of shape: B, V, 3
    # return: image tensor with shape: B, C, H, W
    if mesh.textures is None:
        verts = mesh.verts_list()
        verts_rgb_list = []
        for i in range(len(verts)):
            # print(verts.min(), verts.max())
            verts_rgb_i = torch.ones_like(verts[i])
            if color is not None:
                for i in range(3):
                    verts_rgb_i[:, i] = color[i]
            verts_rgb_list.append(verts_rgb_i)

        texture = pytorch3d.renderer.Textures(verts_rgb=verts_rgb_list)
        mesh.textures = texture

    images = renderer(mesh)
    return images.permute(0, 3, 1, 2)


def sdf_to_mesh(sdf, level=0.02, color=None, render_all=False):
    # device='cuda'
    device = sdf.device

    # extract meshes from sdf
    n_cell = sdf.shape[-1]
    bs, nc = sdf.shape[:2]

    assert nc == 1

    nimg_to_render = bs
    if not render_all:
        if bs > 16:
            cprint('Warning! Will not return all meshes', 'red')
        nimg_to_render = min(bs, 16)  # no need to render that much..

    verts = []
    faces = []
    verts_rgb = []

    for i in range(nimg_to_render):
        sdf_i = sdf[i, 0].detach().cpu().numpy()
        # verts_i, faces_i = mcubes.marching_cubes(sdf_i, 0.02)
        verts_i, faces_i = mc.marching_cubes(sdf_i, level)
        verts_i = verts_i / n_cell - .5
        verts_i[:, [0, 1, 2]] = verts_i[:, [2, 1, 0]]
        rot_func = RotateAxisAngle(-45, "Y", device="cpu")
        verts_i = rot_func.transform_points(
            torch.Tensor(verts_i)).detach().cpu().numpy()
        # verts_i[:, 0] = verts_i[:, 0]*-1
        # verts_i[:, 2] = verts_i[:, 2]*-1
        # verts_i[:, 1] = verts_i[:, 1]*-1
        verts_i = torch.from_numpy(verts_i).float().to(device)
        # verts_i[:, 0] = verts_i[:, 0]*-1
        # verts_i[:, 2] = verts_i[:, 2]*-1
        # verts_i[:, [0, 1, 2]] = verts_i[:, [0, 2, 1]]
        faces_i = torch.from_numpy(faces_i.astype(np.int64)).to(device)
        text_i = torch.ones_like(verts_i).to(device)
        if color is not None:
            for i in range(3):
                text_i[:, i] = color[i]

        verts.append(verts_i)
        faces.append(faces_i)
        verts_rgb.append(text_i)

    try:
        p3d_mesh = pytorch3d.structures.Meshes(
            verts, faces, textures=pytorch3d.renderer.Textures(verts_rgb=verts_rgb))
    except:
        p3d_mesh = None

    return p3d_mesh


def render_sdf(mesh_renderer, sdf, level=0.1, color=None, render_imsize=256, render_all=False):
    """ 
        shape of sdf:
        - bs, 1, nC, nC, nC 

        return a tensor of image rendered according to self.renderer
        shape of image:
        - bs, rendered_imsize, rendered_imsize, 4

        ref: https://github.com/shubhtuls/PixelTransformer/blob/03b65b8612fe583b3e35fc82b446b5503dd7b6bd/data/base_3d.py
    """
    # device='cuda'
    device = sdf.device
    bs = sdf.shape[0]

    if not render_all:
        nimg_to_render = min(bs, 16)  # no need to render that much..

    p3d_mesh = sdf_to_mesh(sdf, level=level, color=color,
                           render_all=render_all)

    if p3d_mesh is not None:
        rendered_im = einops.rearrange(mesh_renderer(
            p3d_mesh), 'b h w c-> b c h w').contiguous()  # bs, h, w, c
    else:
        rendered_im = torch.zeros(
            nimg_to_render, 4, render_imsize, render_imsize).to(device)

    return rendered_im, p3d_mesh


def rotate_mesh(mesh, axis='Y', angle=10, device='cuda'):
    rot_func = RotateAxisAngle(angle, axis, device=device)

    verts = mesh.verts_list()
    faces = mesh.faces_list()
    textures = mesh.textures

    B = len(verts)

    rot_verts = []
    for i in range(B):
        v = rot_func.transform_points(verts[i])
        rot_verts.append(v)
    new_mesh = Meshes(verts=rot_verts, faces=faces, textures=textures)
    return new_mesh


def rotate_mesh_360(mesh_renderer, mesh):
    cur_mesh = mesh

    B = len(mesh.verts_list())
    ret = [[] for i in range(B)]

    for i in range(36):
        cur_mesh = rotate_mesh(cur_mesh)
        # b c h w # important!! no norm here or they will not align
        img = render_mesh(mesh_renderer, cur_mesh, norm=False)
        img = img.permute(0, 2, 3, 1)  # b h w c
        img = img.detach().cpu().numpy()
        img = (img * 255).astype(np.uint8)
        for j in range(B):
            ret[j].append(img[j])

    return ret


def load_mesh(obj_f):
    verts, faces_tup, _ = pytorch3d.io.load_obj(obj_f, load_textures=False)
    faces = faces_tup.verts_idx

    verts = verts.unsqueeze(0)
    faces = faces.unsqueeze(0)

    verts_rgb = torch.ones_like(verts)
    mesh = pytorch3d.structures.Meshes(
        verts=verts, faces=faces, textures=pytorch3d.renderer.TexturesVertex(verts_rgb))

    return mesh


def as_mesh(scene_or_mesh):
    """
    Convert a possible scene to a mesh.
    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        if len(scene_or_mesh.geometry) == 0:
            mesh = None  # empty scene
        else:
            # we lose texture information here
            mesh = trimesh.util.concatenate(
                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                      for g in scene_or_mesh.geometry.values()))
    else:
        assert (isinstance(scene_or_mesh, trimesh.Trimesh))
        mesh = trimesh.Trimesh(
            vertices=scene_or_mesh.vertices, faces=scene_or_mesh.faces)
    return mesh


def get_normalize_mesh(model_file):
    total = 16384
    # print("[*] trimesh_load:", model_file)
    mesh_list = trimesh.load_mesh(model_file, process=False)

    mesh = as_mesh(mesh_list)  # from s2s
    if not isinstance(mesh, list):
        mesh_list = [mesh]

    area_sum = 0
    area_lst = []
    for idx, mesh in enumerate(mesh_list):
        area = np.sum(mesh.area_faces)
        area_lst.append(area)
        area_sum += area
    area_lst = np.asarray(area_lst)
    amount_lst = (area_lst * total / area_sum).astype(np.int32)
    points_all = np.zeros((0, 3), dtype=np.float32)
    for i in range(amount_lst.shape[0]):
        mesh = mesh_list[i]
        # print("start sample surface of ", mesh.faces.shape[0])
        points, index = trimesh.sample.sample_surface(mesh, amount_lst[i])
        # print("end sample surface")
        points_all = np.concatenate([points_all, points], axis=0)
    centroid = np.mean(points_all, axis=0)
    points_all = points_all - centroid
    m = np.max(np.sqrt(np.sum(points_all ** 2, axis=1)))
    # obj_file = os.path.join(norm_mesh_sub_dir, "pc_norm.obj")
    ori_mesh_list = trimesh.load_mesh(model_file, process=False)
    ori_mesh = as_mesh(ori_mesh_list)
    ori_mesh.vertices = (ori_mesh.vertices - centroid) / float(m)
    return ori_mesh, centroid, m


def save_mesh_as_gif(mesh_renderer, mesh, nrow=3, out_name='1.gif'):
    """ save batch of mesh into gif """

    # img_comb = render_mesh(mesh_renderer, mesh, norm=False)

    # rotate
    rot_comb = rotate_mesh_360(mesh_renderer, mesh)  # save the first one

    # gather img into batches
    nimgs = len(rot_comb)
    nrots = len(rot_comb[0])
    H, W, C = rot_comb[0][0].shape
    rot_comb_img = []
    for i in range(nrots):
        img_grid_i = torch.zeros(nimgs, H, W, C)
        for j in range(nimgs):
            img_grid_i[j] = torch.from_numpy(rot_comb[j][i])

        img_grid_i = img_grid_i.permute(0, 3, 1, 2)
        img_grid_i = vutils.make_grid(img_grid_i, nrow=nrow)
        img_grid_i = img_grid_i.permute(1, 2, 0).numpy().astype(np.uint8)

        rot_comb_img.append(img_grid_i)

    with imageio.get_writer(out_name, mode='I', duration=.01) as writer:

        # combine them according to nrow
        for rot in rot_comb_img:
            writer.append_data(rot)

    cprint(f"{out_name} saved", "blue")
