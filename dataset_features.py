# We manage all the idiosyncracies of the original dataset's structure here
import numpy as np
from collections import namedtuple
import mahotas as mt

# name of the feature: index you want in the pandas dataframe
# get_feature: function that takes a sample from a dataset and returns the feature
# countable: boolean that tells us if it makes sense to count the feature
# multiple: boolean that tells us if the feature can simultaneously have multiple values and should be explanded to a one-hot encoding
Feature = namedtuple("feature", ["name", "get_feature", "countable", "multiple"])

def expand_feature(feature, value):
    return f"{feature.name}: \"{value}\""

def get_intensity_stats(sample):
    return (np.mean(sample['image']), np.var(sample['image']))

img_intensity = Feature("Intensity Mean", get_intensity_stats, False, False)


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

img_haralick = Feature("Haralick Features", get_image_haralick, False, False)


# Slows down pandas too much
def get_image_actual(sample):
    return sample['image']

def get_image_hash(sample):
    return hash(get_image_actual(sample).tobytes())

get_image = get_image_hash

image = Feature("Intensity Mean", get_image, False, False)


def get_protein(sample):
    return sample['caption'].split('/')[0]

protein = Feature("Protein", get_protein, True, False) # arguable, but too many proteins to one-hot encode


def get_cell_line(sample):
    return sample['caption'].split('/')[1]

cell_line = Feature("Cell Line", get_cell_line, True, False)


def get_location(sample):
    return sample['caption'].split('/')[2].split(',')

location = Feature("Location", get_location, True, True)
