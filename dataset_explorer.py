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
import json
import umap
import umap.plot
import numpy as np
import prettyprinter as pp
from collections import Counter
import matplotlib.pyplot as plt
from dataset_features import protein, cell_line, location, get_image, img_intensity, img_haralick, expand_feature
from dataset_utils import load_config, allocate_logdir
from dataset_utils import HPAToPandas, get_counts, get_common_features, get_total_features
from dataset_utils import plot_profile


config, opt = load_config()
logdir = allocate_logdir(config, opt)

log_file = os.path.join(logdir, f"dataset_explorer.log")
print(f"Saving logs to {log_file}")
sys.stdout = open(log_file, "w")

features = [protein, cell_line, location, img_intensity, img_haralick]
datasets = HPAToPandas(config, features, expand="total")
counts = get_counts(datasets, features)

for dataset_counts, dataset_name in zip(counts, datasets.keys()):
    plot_profile(features, dataset_counts, dataset_name, logdir)

# calculate counts of common proteins, cell-lines, and localizations between datasets
# for each dataset, get the intersection of proteins, cell-lines, and localizations
# express them as percent shared out of total and for each dataset's set of classes
common = get_common_features(datasets, features, logdir)
total = get_total_features(datasets, features, logdir)
common_counts = {feature: len(vals) for feature, vals in common.items()}
total_counts = {feature: len(vals) for feature, vals in total.items()}
print("\n")
pp.pprint(f"Common features: {common_counts}")
pp.pprint(f"Total features: {total_counts}")

# plot u-map of haralick features with localization as color
# plot only the ones that are common between datasets
# if proteins multilocalize, they can be plotted for either or both locations
# if there aren't enough samples for a location they are sampled with replacement for simplicity
# this is going to over sample multilocalizing proteins
for dataset_name, dataset in datasets.items():
    # subsample locations to have a balanced, manageable UMAP dataset
    location_samples = []
    for loc in common[location.name]:
        try:
            location_samples.extend(
                dataset.loc[
                    lambda df: df[expand_feature(location, loc)], 
                    [location.name, img_haralick.name]
                ].sample(100).values
            )
        except ValueError:
            print(f"Skipping {loc} because there aren't enough samples for it.")

    # convert the multilocalizing proteins to their own class
    loc_hara_list = np.stack(location_samples)
    for i, loc in enumerate(loc_hara_list[:, 0]):
        if len(loc) > 1:
            loc_hara_list[i, 0] = "multilocalizing"
        else:
            loc_hara_list[i, 0] = loc[0]

    # plot the UMAP
    transform = umap.UMAP().fit(np.stack(loc_hara_list[:, 1])) # haralick features
    plt.clf()
    plot = umap.plot.points(transform, labels=loc_hara_list[:, 0])
    plot.get_legend().set(bbox_to_anchor=(1.05, 1.0))
    plt.tight_layout()
    plt.savefig(os.path.join(logdir, f"{dataset_name}_texture_umap.png"))

# for feature in common:
#     prop_shared = len(common[feature]) / len(total[feature])
#     print(f"\nFeature {feature} has {str(round(100 * prop_shared, 2))}% shared across dataset out of {len(total[feature])} total.")
#     for dataset in data_exp.dataset_names():
#         feature_counts = profiles[dataset].get_feature(feature)
#         prop_dataset = len(common[feature]) / len(feature_counts)
#         printTab(f"{dataset} shares {str(round(100 * prop_dataset, 2))}% of {feature} with the rest of the dataset.")

# # report KL-divergence between attribute distributions
# # need ps to get list across all keys, not just those present in one dataset or the other
# div_smoothing = {feature: [] for feature, _ in profile_features}
# smoothing_range = [0.1 ** i for i in range(0, 10)]
# for smoothing in smoothing_range:
#     for feature in div_smoothing:
#         # TODO: way to do this programatically? for train and validation?
#         div_smoothing[feature].append(data_exp.kl_div('validation', 'train', feature, smoothing))

# # plot line graph of KL-divergence per feature as smoothing varies
# plt.clf()
# fig, axs = plt.subplots(1, len(div_smoothing))
# fig.suptitle('KL-divergence per feature as smoothing varies')
# for i, attribute in enumerate(div_smoothing):
#     axs[i].set_title(attribute)
#     axs[i].plot(smoothing_range, div_smoothing[attribute])
#     axs[i].set_xscale('log')
# plt.savefig(os.path.join(logdir, f'kl_divergence.png'))


# # TODO: maybe we should just make lists and filter/match things later
# # like this workflow below doesn't fit easily into the profile structure
# # get the localizations of each protein
# # TODO: could add an option for counts or just dictionary rule or smthg
# protein_localizations = {}
# for dataset, samples in data_exp:
#     for sample in samples:
#         prot = get_protein(sample)[0]
#         loc = get_location(sample)[0]
#         if prot not in protein_localizations:
#             protein_localizations[prot] = set()
#         protein_localizations[prot].update(loc)

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
