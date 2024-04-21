"""PyTorch base data set class each data set takes 2 sets of options a global one and a specific dataset config"""
import torch
from pathlib import Path
from datasets.base_dataset import BaseDataSet
import json
import numpy as np
import os


class BaseShapeNet(BaseDataSet):
    def __init__(self, dataset_options, shape_net_options, cat=None):
        self.cat = shape_net_options["category"] if cat is None else cat
        self.num_classes = 13
        self.class_name_mapping = json.loads(
            Path("datasets/shape_net/shape_info.json").read_text())
        self.category_directory_mapping = json.loads(
            Path("datasets/shape_net/shape_class_info.json").read_text())
        self.classes = sorted(self.class_name_mapping.keys())
        self.class_names = sorted(self.class_name_mapping.values())
        super().__init__(
            dataset_options, shape_net_options)

    def get_category_shape_ids(self, category_id):
        ids = os.listdir(
            self.dataset_path / category_id)
        id_categories = map(lambda id: f"{category_id}/{id}", ids)
        return list(id_categories)

    def is_all_categories(self):
        return self.cat == "all"

    def get_items(self):
        items = []
        if (self.is_all_categories()):
            for category in self.classes:
                shape_ids = self.get_category_shape_ids(category)
                items.extend(shape_ids)
            return items
        category_id = self.category_directory_mapping[self.cat]
        return self.get_category_shape_ids(category_id)
