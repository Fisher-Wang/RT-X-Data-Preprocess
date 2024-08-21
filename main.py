import numpy as np
import rlds
import tensorflow as tf
import tensorflow_datasets as tfds
import yaml
from matplotlib import pyplot as plt
from tqdm import tqdm

from const import DATASETS, VERSIONS
from utils import dataset2path, write_yaml

obs_summary = {}

for dataset in tqdm(DATASETS, desc="Datasets"):
    tqdm.write(f"Processing {dataset}")

    try:
        # Create a dataset object to obtain episode from
        builder = tfds.builder_from_directory(builder_dir=dataset2path(dataset))
        ds = builder.as_dataset(split="train[:1]")
        ds_iterator = iter(ds)

        # Obtain the steps from one episode from the dataset
        episode = next(ds_iterator)
        steps = episode[rlds.STEPS]

        # Obtain the observation keys from the first step
        step = next(iter(steps))
        obs_keys = list(step["observation"].keys())

    except Exception as e:
        tqdm.tqdm.write(f"Error processing {dataset}: {e}")
        obs_keys = None

    obs_summary[dataset] = obs_keys

write_yaml("obs_keys.yaml", obs_summary)
