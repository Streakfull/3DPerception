from src.metrics.chamfer_dist import ChamferDistance
from src.metrics.iou import Iou
from src.metrics.signed_iou import SignedIou


class Metrics():
    def __init__(self, metrics_config):
        self.metrics_config = metrics_config.split(",")
        self.metrics = []

    def get_metrics(self):
        for metric in self.metrics_config:
            match metric:
                case "iou":
                    self.metrics.append(("iou", Iou()))
                case "chamferDistance":
                    self.metrics.append(
                        ("chamferDistance", ChamferDistance(apply_center=True)))
                case "signedIou":
                    self.metrics.append(("signedIou", SignedIou))
                case _:
                    raise Exception("Metric not supported")
        return self.metrics
