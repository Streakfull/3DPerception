from src.utils.utils_3d import init_mesh_renderer, sdf_to_mesh, render_sdf, save_mesh_as_gif
from src.utils.visualizations import tensor2im, save_image
from einops import rearrange


class SDFVisualizer:

    def __init__(self, device, output_path, key, epoch, iteration, logger):
        dist, elev, azim = 1.7, 20, 20
        self.render = init_mesh_renderer(
            image_size=256, dist=dist, elev=elev, azim=azim, device=device)
        self.output_path = output_path
        self.key = key
        self.epoch = epoch
        self.iteration = iteration
        self.logger = logger

    def visualize(self, sdf):
        im, mesh = render_sdf(self.render, sdf)
        self.handle_images(im)
        self.handle_gif(mesh)

    def handle_images(self, images):
        imgs = tensor2im(images)
        save_image(
            imgs, image_path=f"{self.output_path}/{self.key}/epoch_{self.epoch}_iter_{self.iteration}.png")
        images = images[:, 0:3, :, :]
        images = rearrange(images, 'bs ch w h->bs w h ch')
        images = images + 1 / (2.0*255.)
        self.logger.log_image(
            f"Train/{self.key}", images, self.iteration)
 # for i in range(images.shape[0]):
        #     self.logger.log_image(
        #         f"Train/{self.key}", images[i].squeeze(), self.iteration+i)

    def handle_gif(self, mesh):
        try:
            if (mesh is not None):
                save_mesh_as_gif(self.render, mesh,
                                 out_name=f"{self.output_path}/{self.key}/epoch_{self.epoch}_iter_{self.iteration}.gif")
        except:
            import pdb
            pdb.set_trace()
