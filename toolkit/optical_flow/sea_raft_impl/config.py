import json
from collections import namedtuple

CfgNode = namedtuple('CfgNode', ['items'], defaults=[{}])


def load_cfg(path):
    with open(path, 'r') as f:
        cfg_dict = json.load(f)
    return CfgNode(cfg_dict)
