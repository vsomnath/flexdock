import os
import yaml
from omegaconf import OmegaConf

from lightning.pytorch.utilities import rank_zero_only


def read_strings_from_txt(path):
    # every line will be one element of the returned list
    with open(path) as file:
        lines = file.readlines()
        return [line.rstrip() for line in lines]


@rank_zero_only
def save_yaml_file(path, content):
    assert isinstance(
        path, str
    ), f"path must be a string, got {path} which is a {type(path)}"
    content = yaml.dump(data=content)
    if (
        "/" in path
        and os.path.dirname(path)
        and not os.path.exists(os.path.dirname(path))
    ):
        os.makedirs(os.path.dirname(path))
    with open(path, "w") as f:
        f.write(content)


@rank_zero_only
def save_config(cfg, path):
    with open(path, "w") as f:
        OmegaConf.save(cfg, f)
