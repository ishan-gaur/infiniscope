# TODO these need to be abstracted so they can be used over model outputs at inferece time
# TODO use the streaming module instead of loading into memory?

import os
import sys
import csv
import copy
import datetime
import numpy as np
import prettyprinter as pp
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

ckptdir = os.path.join(logdir, "checkpoints")
cfgdir = os.path.join(logdir, "configs")
seed_everything(opt.seed)

# Get config
configs = [OmegaConf.load(cfg) for cfg in opt.base]
cli = OmegaConf.from_dotlist(unknown)
config = OmegaConf.merge(*configs, cli)

# setup logger to send output to file
# TODO: change logdir name to include dataset name
logdir = os.path.join(opt.logdir, f"dataset_{config.data['name']}_{now}")
log_file = os.path.join(logdir, f"dataset_explorer.log")
os.makedirs(os.path.dirname(log_file))
print(f"Saving logs to {log_file}")
sys.stdout = open(log_file, "w")


# Retrieve dataset from config
# NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
# calling these ourselves should not be necessary but it is.
# lightning still takes care of proper multiprocessing though
data = instantiate_from_config(config.data)
data.prepare_data()
data.setup()

# Retrieve protein, cell_line, and location counts
data_profile = {}
start_index = 0
for k in data.datasets:
    print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")
    
    # TODO: All the samples in data are in a single array
    offset = len(data.datasets[k])
    dataset = data.datasets[k].samples[start_index : start_index + offset] 
    start_index += offset
    
    # data_profile is an array of dictionaries, one for each dataset
    data_profile[k] = {}
    proteins, cell_lines, locs = Counter(), Counter(), Counter()

    for sample in dataset:
        sample_type = sample['caption'].split('/')
        sample_type[2] = sample_type[2].split(',')
        # TODO: check that this isn't actually used
        if sample_type[0] == 'nan':
            continue
        proteins.update([sample_type[0]])
        cell_lines.update([sample_type[1]])
        locs.update(sample_type[2])

    data_profile[k]["Protein Counts"] = proteins
    data_profile[k]["Cell Line Counts"] = cell_lines
    data_profile[k]["Localization Counts"] = locs

    # plot and save data
    fig, axs = plt.subplots(1, len(data_profile[k]))
    for i, label in enumerate(data_profile[k].keys()):
        x = data_profile[k][label].values()
        logbins = np.logspace(np.log(min(x)), np.log(max(x)), 10)
        axs[i].hist(x, bins=10, density=True)
        axs[i].title.set_text(label)

        with open(os.path.join(logdir, f"{k}_{label}.csv"), "w", newline="") as f:
            fieldnames = [label, "count"]
            writer = csv.writer(f)
            writer.writerow(fieldnames)
            for key, val in data_profile[k][label].items():
                writer.writerow([key, val])   

    plt.savefig(os.path.join(logdir, f'{k}.png'))

# calculate counts of common proteins, cell-lines, and localizations between datasets
# for each dataset, get the intersection of proteins, cell-lines, and localizations
common = copy.deepcopy(data_profile[k])
for k in data_profile.keys():
    for c in common:
        common[c] = common[c] & data_profile[k][c]

for c in common:
    for v in common[c]:
        # TODO: denominator should be updated to exclude nans
        common[c][v] = [common[c][v] / len(data.datasets[k]) for k in data_profile.keys()]

# convert counts to percentages of each dataset
for label in common:
    for k, counts in common[label].items():
        common[label][k] = [str(round(100 * count, 1)) + "%" for count in counts]

pp.pprint(common)

# get the localizations of each protein
protein_localizations = {}
start_index = 0
for k in data.datasets:
    offset = len(data.datasets[k])
    dataset = data.datasets[k].samples[start_index : start_index + offset] # TODO: All the samples in data are in a single array
    start_index += offset
    
    for sample in dataset:
        sample_type = sample['caption'].split('/')
        sample_type[2] = sample_type[2].split(',')
        if sample_type[0] not in protein_localizations:
            protein_localizations[sample_type[0]] = set()
        protein_localizations[sample_type[0]].update(sample_type[2])

# count the number of localizations for each protein
protein_localization_counts = {}
for protein, localizations in protein_localizations.items():
    protein_localization_counts[protein] = len(localizations)

# plot counts of localizations for each protein
plt.clf()
x = protein_localization_counts.values()
plt.hist(x, bins=10)
plt.title("Protein Localization Counts")
plt.savefig(os.path.join(logdir, 'protein_localization_counts.png'))

# print protein localization counts by dataset
for k in data_profile:
    multilocalizing = 0
    for protein in data_profile[k]["Protein Counts"]:
        if protein_localization_counts[protein] > 1:
            multilocalizing += 1
    print(f"{k} total proteins {len(data_profile[k]['Protein Counts'])}")
    print(f"{k} has {multilocalizing} multilocalizing proteins")

# histograms of dataset image intensities
plt.clf()
fig, axs = plt.subplots(2, len(data.datasets), sharey=True)
start_index = 0
for i, k in enumerate(data.datasets):
    offset = len(data.datasets[k])
    dataset = data.datasets[k].samples[start_index : start_index + offset] # TODO
    start_index += offset

    mean_intensities = []
    var_intensities = []
    for sample in dataset:
        mean_intensities.append(sample['image'].mean())
        var_intensities.append(sample['image'].var())

    axs[0][i].hist(mean_intensities, bins=10, density=True)
    axs[0][i].title.set_text(f"{k} Intensity Mean Histogram")
    axs[1][i].hist(var_intensities, bins=10, density=True)
    # axs[1][i].title.set_text(f"{k} Intensity Var Histogram")

plt.savefig(os.path.join(logdir, 'intensity_histograms.png'))