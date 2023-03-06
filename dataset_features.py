# We manage all the idiosyncracies of the original dataset's structure here
import os
import numpy as np
from collections import namedtuple
import mahotas as mt
import pandas as pd
import hashlib
from collections.abc import Iterable

# name of the feature: index you want in the pandas dataframe
# get_feature: function that takes a sample from a dataset and returns the feature
# countable: boolean that tells us if it makes sense to count the feature
# multiple: boolean that tells us if the feature can simultaneously have multiple values and should be explanded to a one-hot encoding
Feature = namedtuple("feature", ["name", "get_feature", "countable", "multiple"])

def expand_feature(feature, value):
    return f"{feature.name}: \"{value}\""

def get_intensity_mean(sample):
    return np.mean(sample['image'])

def get_intensity_var(sample):
    return np.var(sample['image'])

def normalized_to_uint8(image):
    image -= np.min(image)
    image /= np.max(image)
    image *= 255
    return image.astype(np.uint8)

def get_image_haralick(sample):
    # angular second moment, contrast, correlation, sum of squares: variance, 
    # inverse difference moment, sum average, sum variance, sum entropy, entropy, 
    # difference variance, difference entropy, information measures of correlation 1, 
    # information measures of correlation 2
    image = normalized_to_uint8(sample['image'])
    textures = mt.features.haralick(image) # 13 arrays since 13 3D directions (not 14?)
    ht_mean = textures.mean(axis=0)
    return ht_mean

# Slows down pandas too much
def get_image(sample):
    return sample['image']

def get_image_hash(sample):
    image_hash = hashlib.sha1(get_image(sample).tobytes()).hexdigest()
    return image_hash

def get_protein(sample):
    return sample['caption'].split('/')[0]

def get_cell_line(sample):
    return sample['caption'].split('/')[1]

def get_location(sample):
    return sample['caption'].split('/')[2].split(',')

image = Feature("Image", get_image, False, False)
img_hash = Feature("Image Hash", get_image_hash, False, False)
int_mean = Feature("Intensity Mean", get_intensity_mean, False, False)
int_var = Feature("Intensity Var", get_intensity_var, False, False)
img_haralick = Feature("Haralick Features", get_image_haralick, False, False)
protein = Feature("Protein", get_protein, True, False) # arguable, but too many proteins to one-hot encode
cell_line = Feature("Cell Line", get_cell_line, True, False)
location = Feature("Location", get_location, True, True)

# TODO: untested as far as missing/features
# Tested on missing images from the cache
# TODO: untested on subset of features from cache entry
# Tries to find sample in cache
# If it's not there it computes them all and then adds an entry
# If it's missing features for that image, it computes them and adds them to the entry
# Always returns a complete list of features in the order of the features list
# TODO: needs a command to refresh cache, for example if feature code was wrong
# Just maintains a picked pandas dataframe so loading speed/size might become a bottle neck
# If it really gets too big, we would need to use a database and expose it as a service maybe
# TODO: do all samples together
def get_features(samples, features, cache, logdir):
    if not cache:
        return [{f.name: f.get_feature(sample) for f in features} for sample in samples]

    if logdir is None:
        raise ValueError("logdir must be specified if cache is True")

    if img_hash not in features:
        raise ValueError("Image hash must be in features to lookup cache entries.")

    cache_path = os.path.join(logdir, "feature_cache.pkl")

    df = pd.DataFrame(columns=[f.name for f in features])
    if os.path.exists(cache_path):
        df = pd.read_pickle(cache_path)

    sample_features = []
    changed = False
    for sample in samples:
        feature_values, sample_changed, df = retrieve_from_cache(sample, features, df)
        sample_features.append(feature_values)
        changed |= sample_changed

    if changed:
        with open(cache_path, "wb") as f:
            df.to_pickle(f)

    return sample_features


def retrieve_from_cache(sample, features, df):
    feature_values = {}
    key = img_hash.get_feature(sample)
    changed = False
    if key not in df[img_hash.name].values:
        feature_values = {f.name: f.get_feature(sample) for f in features}
        df = df.append(feature_values, ignore_index=True)
        changed = True
    else:
        for f in features:
            if f.name not in df.columns:
                feature_values[f.name] = f.get_feature(sample)
                changed = True
            else:
                val = df.loc[df[img_hash.name] == key, f.name].values[0]
                use_any = type(val) in [np.ndarray, list]
                if (use_any and not pd.isna(val).any()) or (not use_any and not pd.isna(val)):
                    feature_values[f.name] = val
                else:
                    feature_values[f.name] = f.get_feature(sample)
                    changed = True

            # the below will account for new columns too
            if changed == True:
                df.drop(df[img_hash.name] == key, inplace=True)
                df = df.append(feature_values, ignore_index=True)

    return feature_values, changed, df
