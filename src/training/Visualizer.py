
from termcolor import cprint
from src.training.SDFVisualizer import SDFVisualizer
from src.utils.util import mkdir


class Visualizer:

    def __init__(self, device, output_path):
        self.output_path = output_path
        self.device = device
        self.directories_created = False

    def visualize(self, sdf_visuals_dict, epoch, iteration):
        cprint("Running Visualizations", "blue")
        for key in sdf_visuals_dict.keys():
            sdf = sdf_visuals_dict[key]
            if not self.directories_created:
                mkdir(f"{self.output_path}/{key}")
            vis = SDFVisualizer(device=self.device, output_path=self.output_path,
                                key=key, epoch=epoch, iteration=iteration)
            vis.visualize(sdf)

        self.directories_created = True
