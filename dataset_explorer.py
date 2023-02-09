# TODO rerun analysis from calculated stats on existing folder
    # Need to save indices of each dataset for thi
# TODO these need to be abstracted so they can be used over model outputs at inferece time
# TODO and so that they all just deal with the dataset structure or external defined datatypes
# TODO use the streaming module instead of loading into memory?
# TODO protein names contain a list of them and their isoforms, so they are brittle to lookup by other names

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
from dataset_utils import printTab


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
if opt.dev:
    logdir = os.path.join(opt.logdir, "dataset_explorer_dev")
else:
    logdir = os.path.join(opt.logdir, f"dataset_{config.data['name']}_{now}")
log_file = os.path.join(logdir, f"dataset_explorer.log")
os.makedirs(os.path.dirname(log_file), exist_ok=opt.dev)
if opt.dev:
    for root, dirs, files in os.walk(logdir):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))

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
    print(f"\n{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")
    
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

    make_percent = lambda counts: {k: f"{round(100 * v / len(dataset), 2)}%" for (k, v) in counts}
    printTab(f"Top 5 Proteins: {make_percent(proteins.most_common(5))}")
    printTab(f"Top 5 Cell Lines: {make_percent(cell_lines.most_common(5))}")
    printTab(f"Top 5 Locations: {make_percent(locs.most_common(5))}")

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
total = {_k: set(v.keys()) for _k, v in data_profile[k].items()}
for k in data_profile.keys(): # for each dataset
    for c in common: # count of proteins, cell-lines, and localizations
        common[c] = common[c] & data_profile[k][c]
        total[c] = total[c] | set(data_profile[k][c].keys())

for c in common:
    prop_shared = common[c] / total[c]
    print(f"{common[c]} of {total[c]} classes in {c}, or {str(round(100 * prop_shared, 2))}: {len(common[c])}")
    for v in common[c]:
        # TODO: denominator should be updated to exclude nans
        common[c][v] = [common[c][v] / len(data.datasets[k]) for k in data_profile.keys()]

# convert counts to percentages of each dataset
for label in common:
    for k, counts in common[label].items():
        common[label][k] = [str(round(100 * count, 1)) + "%" for count in counts]

pp.pprint(common)

# report KL-divergence between attribute distributions
# need ps to get list across all keys, not just those present in one dataset or the other
div_smoothing = {attr: [] for attr in total}
smoothing_range = [0.1 ** i for i in range(0, 10)]
for smoothing in smoothing_range:
    ps = {k: {attr: [] for attr in data_profile[k]} for k in data_profile}
    # todo sep KL by attribute
    for k in data.datasets:
        for attribute in total:
            for classname in total[attribute]:
                ps[k][attribute].append(data_profile[k][attribute][classname])
            ps[k][attribute] = np.asarray(ps[k][attribute]) + len(data.datasets[k]) * smoothing
            norm = np.sum(ps[k][attribute])
            ps[k][attribute] /= norm 

    combos = list(itertools.combinations(data.datasets.keys(), 2))
    for combo in combos:
        if smoothing == smoothing_range[2]: # turns out the exponent op on floats is imprecise
            print(f"KL-divergence per feature for {combo[1]} and {combo[0]}:")
        for attribute in total: # TODO: save these key lists into specially named things
            div = np.sum(kl_div(ps[combo[1]][attribute], ps[combo[0]][attribute]))
            div_smoothing[attribute].append(div) # only while two datasets
            if smoothing == smoothing_range[2]:
                print(f"\t{attribute}: {div}")

# plot line graph of KL-divergence per feature as smoothing varies
plt.clf()
fig, axs = plt.subplots(1, len(div_smoothing))
fig.suptitle('KL-divergence per feature as smoothing varies')
for i, attribute in enumerate(div_smoothing):
    axs[i].set_title(attribute)
    axs[i].plot(smoothing_range, div_smoothing[attribute])
    axs[i].set_xscale('log')
plt.savefig(os.path.join(logdir, f'kl_divergence.png'))


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
def median(x): return np.median(x)
def mean(x): return np.mean(x)
def var(x): return np.var(x)
stats = [median, mean, var]
for stat in stats:
    plt.clf()
    fig, axs = plt.subplots(1, len(data.datasets), sharex=True, sharey=True)
    fig.suptitle(f'{stat.__name__} Intensity Histograms (PMF)')
    start_index = 0
    for i, k in enumerate(data.datasets):
        offset = len(data.datasets[k])
        dataset = data.datasets[k].samples[start_index : start_index + offset] # TODO
        start_index += offset

        intensities = []
        for sample in dataset:
            intensities.append(stat(sample['image']))

        intensities = np.asarray(intensities)
        intensities = (intensities + 1) / 2 # normalize to [0, 1]
        axs[i].hist(intensities, bins=10, density=True)
        axs[i].title.set_text(k)

    plt.savefig(os.path.join(logdir, f'{stat.__name__}_intensity_histograms.png'))
