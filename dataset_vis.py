import os
import umap
import umap.plot
import numpy as np
import matplotlib.pyplot as plt
from dataset_features import location, img_haralick, expand_feature


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

