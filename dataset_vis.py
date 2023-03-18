import os
import umap
import umap.plot
import numpy as np
import matplotlib.pyplot as plt
from dataset_features import location, img_haralick, expand_feature
from dataset_utils import get_common_features
from scipy.special import kl_div
import seaborn as sns
import pandas as pd
import math


def plot_haralick_umap(dataset, dataset_name, locations, logdir, sample_size=100):
    """Plot UMAP embedding of dataset features."""
    # plot u-map of haralick features with localization as color
    # plot only the ones that are common between datasets
    # if proteins multilocalize, they can be plotted for either or both locations
    # if there aren't enough samples for a location they are sampled with replacement for simplicity
    # this is going to over sampling multilocalizing proteins
    # but we're trying to sample over a given protein, in which case it may be fine
    # subsample locations to have a balanced, manageable UMAP dataset
    if img_haralick.name not in dataset.columns:
        raise ValueError(f"Need haralick features in dataset for UMAP.")

    location_samples = []
    for loc in locations:
        try:
            location_samples.extend(
                dataset.loc[
                    lambda df: df[expand_feature(location, loc)], 
                    [location.name, img_haralick.name]
                ].sample(sample_size).values
            )
        except ValueError:
            print(f"Skipping {loc} because there aren't enough samples for it.")

    if len(location_samples) == 0:
        print("No samples for any location. Skipping UMAP.")
        return

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


# Plot histograms of the features requested from the counts
def plot_profile(features, counts, filename, logdir):
    plt.clf()
    # features = [f for f in features if f.countable]
    features = list(filter(lambda f: f.countable, features))
    fig, axs = plt.subplots(1, len(features))
    fig.suptitle('Histograms of Feature Counts')
    for i, feature in enumerate(features):
        axs[i].hist(counts[feature.name].values(), bins=10, density=True)
        axs[i].title.set_text(feature.name)
    plt.savefig(os.path.join(logdir, f'{filename}.png'))


def plot_conditional_dist(pred_feature, cond_feature, datasets, source_data, target_data,
                          logdir, smoothing=1, summary='histogram', bins=5):
    common = get_common_features({source_data: datasets[source_data], target_data: datasets[target_data]}, [cond_feature])
    kl_divs = {pred_feature.name:[], "kl_div": []}
    Ps, Qs = [], []
    df = pd.DataFrame()
    for feature in common[cond_feature.name]:
        # combine the lists of pred_feature values for the current feature
        Q = np.concatenate(
            datasets[source_data][
                datasets[source_data][cond_feature.name] == feature
            ][pred_feature.name].values.tolist()
        )
        P = np.concatenate(
            datasets[target_data][
                datasets[target_data][cond_feature.name] == feature
            ][pred_feature.name].values.tolist()
        )
        if Q.shape[0] < 100:
            continue
        df = df.append([{cond_feature.name: feature, "dataset": source_data, pred_feature.name: q} for q in Q[:, 0]])
        df = df.append([{cond_feature.name: feature, "dataset": target_data, pred_feature.name: p} for p in P[:, 0]])
        Qs.append(Q)
        Ps.append(P)
        # get 2D histogram counts of the values
        # TODO: this needs to be generalized based on the dimensionality of the data
        # worst case might have to do all of this through a umap for the summary??
        # the bucketing should also be a customizable parameter
        # TODO: use np.histogramdd and have a user defined prior
        # TODO: maybe these functions can be separated
        dims = 2
        smooth_x = np.linspace(0, 1, bins)
        smooth_y = 1 - smooth_x
        smooth_x = np.append(np.concatenate([smooth_x] * smoothing), [0] * smoothing)
        smooth_y = np.append(np.concatenate([smooth_y] * smoothing), [0] * smoothing)
        # sample elements of P and Q to get a better estimate of the KL divergence
        sample_size = 64
        for _ in range(30):
            # Px, Py = np.concatenate([P[:, 0], smooth_x]), np.concatenate([P[:, 1], smooth_y])
            P_sample = P[np.random.choice(P.shape[0], size=sample_size, replace=True)]
            Q_sample = Q[np.random.choice(Q.shape[0], size=sample_size, replace=True)]
            Px, Py = np.concatenate([P_sample[:, 0], smooth_x]), np.concatenate([P_sample[:, 1], smooth_y])
            P_hist, _, _ = np.histogram2d(Px, Py, bins=5, density=True)
            P_hist /= P_hist.sum()
            # Qx, Qy = np.concatenate([Q[:, 0], smooth_x]), np.concatenate([Q[:, 1], smooth_y])
            Qx, Qy = np.concatenate([Q_sample[:, 0], smooth_x]), np.concatenate([Q_sample[:, 1], smooth_y])
            Q_hist, _, _ = np.histogram2d(Qx, Qy, bins=5, density=True)
            Q_hist /= Q_hist.sum()
            # compute kl divergence
            kl = np.sum(kl_div(P_hist, Q_hist))
            if math.isinf(kl):
                continue
            kl_divs[pred_feature.name].append(feature)
            kl_divs["kl_div"].append(kl)
    kl_df = pd.DataFrame(kl_divs)

    plt.clf()
    if summary == 'scatter':
        return NotImplementedError("Scatter plots not implemented yet.")
    elif summary == 'histogram':
        plt.clf()
        sns.violinplot(data=df, x=cond_feature.name, y=pred_feature.name, hue="dataset", split=True)
        plt.tight_layout()
    plt.savefig(os.path.join(logdir, f'Q={source_data},P={target_data}.png'))

    # plot bar chart from kl_divs dictionary
    plt.clf()
    sns.barplot(data=kl_df, x=pred_feature.name, y="kl_div")
    plt.tight_layout()
    plt.savefig(os.path.join(logdir, f'Q={source_data},P={target_data}_{pred_feature.name}|{cond_feature.name}_KL_Divs.png'))

    return common