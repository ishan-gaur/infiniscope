# TODO rerun analysis from calculated stats on existing folder
    # Need to save indices of each dataset for thi
# TODO timestamp data_caches so that the Explorer class can lazily regenerate stuff
# TODO these need to be abstracted so they can be used over model outputs at inferece time
# TODO and so that they all just deal with the dataset structure or external defined datatypes
# TODO use the streaming module instead of loading into memory?
# TODO protein names contain a list of them and their isoforms, so they are brittle to lookup by other names
# TODO add metrics on the other input data as well? Like for the cell line and protein encoding?

# TODO: Classes for datasets with methods to analyze them
# TODO: Classes for datasets that are constructed using another one have constructors made accordingly

import os
import sys
import csv
import copy
import itertools
import numpy as np
import prettyprinter as pp
from collections import Counter
import matplotlib.pyplot as plt
from scipy.special import kl_div
from pytorch_lightning.trainer import Trainer

from dataset_utils import DatasetExplorer, DataProfile
from dataset_utils import load_config, allocate_logdir, printTab


config, opt = load_config()
logdir = allocate_logdir(config, opt)

log_file = os.path.join(logdir, f"dataset_explorer.log")
print(f"Saving logs to {log_file}")
sys.stdout = open(log_file, "w")

data_exp = DatasetExplorer(config, logdir)

# See reference of DataProfile class for more details
# We need to pass in a list of tuples, where each tuple is a label 
# and a function that extracts the feature from a sample and returns
# a list of values to add to the counter.
def get_protein(sample):
    return [sample['caption'].split('/')[0]]

def get_cell_line(sample):
    return [sample['caption'].split('/')[1]]

def get_location(sample):
    return sample['caption'].split('/')[2].split(',')

PROT, CELL, LOC = "Protein", "Cell Line", "Location"
profile_features = [
    (PROT, get_protein),
    (CELL, get_cell_line),
    (LOC, get_location),
]

# Retrieve protein, cell_line, and location counts
profiles = data_exp.get_profile(profile_features)
for dataset, profile in profiles.items():
    print(f"\nDataset: {dataset}")
    for feature, _ in profile_features:
        printTab(f"Top 5 {feature}s: {profile.top_percents(feature)}")

    profile.plot_profile(profile_features, dataset)
    profile.write_to_csv(profile_features, dataset)


# # calculate counts of common proteins, cell-lines, and localizations between datasets
# # for each dataset, get the intersection of proteins, cell-lines, and localizations
# common = copy.deepcopy(data_profile[dataset_name])
# total = {_k: set(v.keys()) for _k, v in data_profile[dataset_name].items()}
# for k in data_profile.keys(): # for each dataset
#     for c in common: # count of proteins, cell-lines, and localizations
#         common[c] = common[c] & data_profile[k][c]
#         total[c] = total[c] | set(data_profile[k][c].keys())

# for c in common:
#     prop_shared = len(common[c]) / len(total[c])
#     print(f"{common[c]} of {total[c]} classes in {c}, or {str(round(100 * prop_shared, 2))}: {len(common[c])}")
#     for v in common[c]:
#         # TODO: denominator should be updated to exclude nans
#         common[c][v] = [common[c][v] / len(data.datasets[k]) for k in data_profile.keys()]

# # convert counts to percentages of each dataset
# for label in common:
#     for k, counts in common[label].items():
#         common[label][k] = [str(round(100 * count, 1)) + "%" for count in counts]

# pp.pprint(common)

# # report KL-divergence between attribute distributions
# # need ps to get list across all keys, not just those present in one dataset or the other
# div_smoothing = {attr: [] for attr in total}
# smoothing_range = [0.1 ** i for i in range(0, 10)]
# for smoothing in smoothing_range:
#     ps = {k: {attr: [] for attr in data_profile[k]} for k in data_profile}
#     # todo sep KL by attribute
#     for k in data.datasets:
#         for attribute in total:
#             for classname in total[attribute]:
#                 ps[k][attribute].append(data_profile[k][attribute][classname])
#             ps[k][attribute] = np.asarray(ps[k][attribute]) + len(data.datasets[k]) * smoothing
#             norm = np.sum(ps[k][attribute])
#             ps[k][attribute] /= norm 

#     combos = list(itertools.combinations(data.datasets.keys(), 2))
#     for combo in combos:
#         if smoothing == smoothing_range[2]: # turns out the exponent op on floats is imprecise
#             print(f"KL-divergence per feature for {combo[1]} and {combo[0]}:")
#         for attribute in total: # TODO: save these key lists into specially named things
#             div = np.sum(kl_div(ps[combo[1]][attribute], ps[combo[0]][attribute]))
#             div_smoothing[attribute].append(div) # only while two datasets
#             if smoothing == smoothing_range[2]:
#                 print(f"\t{attribute}: {div}")

# # plot line graph of KL-divergence per feature as smoothing varies
# plt.clf()
# fig, axs = plt.subplots(1, len(div_smoothing))
# fig.suptitle('KL-divergence per feature as smoothing varies')
# for i, attribute in enumerate(div_smoothing):
#     axs[i].set_title(attribute)
#     axs[i].plot(smoothing_range, div_smoothing[attribute])
#     axs[i].set_xscale('log')
# plt.savefig(os.path.join(logdir, f'kl_divergence.png'))


# # get the localizations of each protein
# protein_localizations = {}
# start_index = 0
# for k in data.datasets:
#     offset = len(data.datasets[k])
#     dataset = data.datasets[k].samples[start_index : start_index + offset] # TODO: All the samples in data are in a single array
#     start_index += offset
    
#     for sample in dataset:
#         sample_type = sample['caption'].split('/')
#         sample_type[2] = sample_type[2].split(',')
#         if sample_type[0] not in protein_localizations:
#             protein_localizations[sample_type[0]] = set()
#         protein_localizations[sample_type[0]].update(sample_type[2])

# # count the number of localizations for each protein
# protein_localization_counts = {}
# for protein, localizations in protein_localizations.items():
#     protein_localization_counts[protein] = len(localizations)

# # plot counts of localizations for each protein
# plt.clf()
# x = protein_localization_counts.values()
# plt.hist(x, bins=10)
# plt.title("Protein Localization Counts")
# plt.savefig(os.path.join(logdir, 'protein_localization_counts.png'))

# # print protein localization counts by dataset
# for k in data_profile:
#     multilocalizing = 0
#     for protein in data_profile[k]["Protein Counts"]:
#         if protein_localization_counts[protein] > 1:
#             multilocalizing += 1
#     print(f"{k} total proteins {len(data_profile[k]['Protein Counts'])}")
#     print(f"{k} has {multilocalizing} multilocalizing proteins")

# # histograms of dataset image intensities
# def median(x): return np.median(x)
# def mean(x): return np.mean(x)
# def var(x): return np.var(x)
# stats = [median, mean, var]
# for stat in stats:
#     plt.clf()
#     fig, axs = plt.subplots(1, len(data.datasets), sharex=True, sharey=True)
#     fig.suptitle(f'{stat.__name__} Intensity Histograms (PMF)')
#     start_index = 0
#     for i, k in enumerate(data.datasets):
#         offset = len(data.datasets[k])
#         dataset = data.datasets[k].samples[start_index : start_index + offset] # TODO
#         start_index += offset

#         intensities = []
#         for sample in dataset:
#             intensities.append(stat(sample['image']))

#         intensities = np.asarray(intensities)
#         intensities = (intensities + 1) / 2 # normalize to [0, 1]
#         axs[i].hist(intensities, bins=10, density=True)
#         axs[i].title.set_text(k)

#     plt.savefig(os.path.join(logdir, f'{stat.__name__}_intensity_histograms.png'))
