# TODO use the streaming module instead of loading into memory?

import os
import sys
import csv
import datetime
import numpy as np
from collections import Counter
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer

from main import get_parser, instantiate_from_config

now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

# add cwd for convenience
sys.path.append(os.getcwd())

parser = get_parser()
parser = Trainer.add_argparse_args(parser)

opt, unknown = parser.parse_known_args()

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

ckptdir = os.path.join(logdir, "checkpoints")
cfgdir = os.path.join(logdir, "configs")
seed_everything(opt.seed)

# try:
    # import pdb; pdb.set_trace()
    # init and save configs
configs = [OmegaConf.load(cfg) for cfg in opt.base]
cli = OmegaConf.from_dotlist(unknown)
config = OmegaConf.merge(*configs, cli)


data = instantiate_from_config(config.data)
# NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
# calling these ourselves should not be necessary but it is.
# lightning still takes care of proper multiprocessing though
data.prepare_data()
data.setup()

data_profile = {}
for k in data.datasets:
    print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")
    dataset = data.datasets[k].samples[:len(data.datasets[k])] # TODO: All the samples in data are in a single array
    data_profile[k] = {}
    proteins, cell_lines, locs = Counter(), Counter(), Counter()

    for sample in dataset:
        sample_type = sample['caption'].split('/')
        sample_type[2] = sample_type[2].split(',')
        proteins.update([sample_type[0]])
        cell_lines.update([sample_type[1]])
        locs.update(sample_type[2])
    
    data_profile[k]["Protein Counts"] = proteins
    data_profile[k]["Cell Line Counts"] = cell_lines
    data_profile[k]["Localization Counts"] = locs

    fig, axs = plt.subplots(1, 3)
    for i, label in enumerate(data_profile[k].keys()):
        x = data_profile[k][label].values()
        logbins = np.geomspace(min(x), max(x), 10)
        axs[i].hist(x, bins=logbins)
        axs[i].title.set_text(label)
        axs[i].set_xscale('log')

        with open(f"{k}_{label}.csv", "w", newline="") as f:
            fieldnames = [label, "count"]
            writer = csv.writer(f)
            writer.writerow(fieldnames)
            for key, val in data_profile[k][label].items():
                writer.writerow([key, val])   

    plt.savefig(f'{k}.png')