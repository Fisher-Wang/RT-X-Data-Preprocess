import yaml

from const import DATASETS, VERSIONS


def write_yaml(path, data):
    with open(path, "w+") as f:
        yaml.dump(data, f, default_flow_style=False)


def dataset2path(dataset_name):
    assert dataset_name in DATASETS, f"Dataset {dataset_name} not found"
    if dataset_name in VERSIONS:
        version = VERSIONS[dataset_name]
    else:
        version = "0.1.0"
    return f"gs://gresearch/robotics/{dataset_name}/{version}"
