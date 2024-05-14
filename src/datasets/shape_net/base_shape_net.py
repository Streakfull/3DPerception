"""PyTorch base data set class each data set takes 2 sets of options a global one and a specific dataset config"""
import json
import os
from pathlib import Path

from src.datasets.base_dataset import BaseDataSet


class BaseShapeNet(BaseDataSet):
    def __init__(self, dataset_options, shape_net_options, cat=None):
        self.cat = shape_net_options.get(
            "category", "all") if cat is None else cat
        self.num_classes = 13
        self.class_name_mapping = json.loads(
            Path("src/datasets/shape_net/shape_info.json").read_text())
        self.category_directory_mapping = json.loads(
            Path("src/datasets/shape_net/shape_class_info.json").read_text())
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

    def get_shape_info(self, shape_key):
        return shape_key.split("/")

    def __getitem__(self, index):
        shape_key = self.items[index]
        shape_info = self.get_shape_info(shape_key)
        class_name = self.class_name_mapping[shape_info[0]]
        class_index = self.classes.index(shape_info[0])
        return shape_key, class_index, class_name, shape_key
