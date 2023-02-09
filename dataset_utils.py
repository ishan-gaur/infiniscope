import os
import sys
import csv
import copy
import shutil
import datetime
import itertools
import numpy as np
import prettyprinter as pp
from collections import Counter
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from scipy.special import kl_div
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer

from main import get_parser, instantiate_from_config


def load_config():
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()

    # setup logging environment
    if opt.name:
        name = "_" + opt.name
    elif opt.base:
        cfg_fname = os.path.split(opt.base[0])[-1]
        cfg_name = os.path.splitext(cfg_fname)[0]
        name = "_" + cfg_name
    else:
        name = ""

    nowname = now + name + opt.postfix
    logdir = os.path.join(opt.logdir, nowname)

    seed_everything(opt.seed)

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    return config, opt, logdir

def printTab(*args):
    args = ("\t",)+args
    print(*args)