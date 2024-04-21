import omegaconf

class BaseOptions():
    def __init__(self,configs_path="./configs/global_configs.yaml"):
        configs = omegaconf.OmegaConf.load(configs_path)
        self.__dict__.update(configs)
