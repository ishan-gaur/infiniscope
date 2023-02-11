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
    # add cwd for convenience
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()

    seed_everything(opt.seed)

    configs = [OmegaConf.load(cfg) for cfg in opt.base]
    cli = OmegaConf.from_dotlist(unknown)
    config = OmegaConf.merge(*configs, cli)

    return config, opt


def allocate_logdir(config, opt):
    config, opt = load_config()

    # setup logging environment
    if opt.name:
        name = "_" + opt.name
    elif opt.base:
        cfg_fname = os.path.split(opt.base[0])[-1]
        cfg_name = os.path.splitext(cfg_fname)[0]
        name = "_" + cfg_name
    else:
        name = ""

    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    nowname = now + name + opt.postfix
    logdir = os.path.join(opt.logdir, nowname)
    # setup logger to send output to file
    if opt.dev:
        logdir = os.path.join(opt.logdir, "dataset_explorer_dev")
    else:
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        logdir = os.path.join(opt.logdir, f"dataset_{config.data['name']}_{now}")

    os.makedirs(os.path.dirname(logdir), exist_ok=opt.dev)
    if opt.dev:
        for root, dirs, files in os.walk(logdir):
            for f in files:
                os.unlink(os.path.join(root, f))
            for d in dirs:
                shutil.rmtree(os.path.join(root, d))

    return logdir

def printTab(*args):
    args = ("\t",)+args
    print(*args)


class DatasetExplorer:
    def __init__(self, config, logdir):
        # Retrieve dataset from config
        # NOTE according to https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html
        # calling these ourselves should not be necessary but it is.
        # lightning still takes care of proper multiprocessing though
        self.data = instantiate_from_config(config.data)
        self.data.prepare_data()
        self.data.setup()
        self.logdir = logdir
        self.profiles = None
        self.features = None
    
    def __iter__(self):
        return DatasetIterator(self.data.datasets)

    # returns list[dataset_names] to index into profiles
    def dataset_names(self):
        return list(self.data.datasets.keys())

    # returns list[DatasetProfile]
    def get_profiles(self, features):
        self.features = [feature_name for feature_name, _ in features]
        self.profiles = {}
        for dataset_name, samples in self:
            class_name = self.data.datasets[dataset_name].__class__.__name__
            print(f"{dataset_name}, {class_name}, {len(samples)}")
            self.profiles[dataset_name] = FeatureCountProfile(samples, features, self.logdir)
        return self.profiles
    
    # returns dict{features : set(labels)}
    def get_common_labels(self):
        if self.profiles is None:
            raise ValueError("Must call get_profiles first")
        common = {}
        for feature in self.features:
            feature_labels = [set(profile.label_list(feature)) for profile in self.profiles.values()]
            common[feature] = set.intersection(*feature_labels)
        return common
    
    # returns dict{features : set(labels)}
    def get_total_labels(self):
        if self.profiles is None:
            raise ValueError("Must call get_profiles first")
        total = {}
        for feature in self.features:
            feature_labels = [set(profile.label_list(feature)) for profile in self.profiles.values()]
            total[feature] = set.union(*feature_labels)
        return total

    def get_feature_pmf(self, dataset, feature, labels, smoothing):
        counts = [self.profiles[dataset].get_feature(feature)[label] for label in labels[feature]]
        pmf = np.asarray(counts) + smoothing * len(self.data.datasets[dataset])
        norm_pmf = pmf / np.sum(pmf)
        return norm_pmf
        

    def kl_div(self, dataset_p, dataset_q, feature, smoothing):
        labels = self.get_total_labels()
        ps = self.get_feature_pmf(dataset_p, feature, labels, smoothing)
        qs = self.get_feature_pmf(dataset_q, feature, labels, smoothing)
        return np.sum(kl_div(ps, qs))

class DatasetIterator:
    def __init__(self, datasets):
        self.datasets = datasets
        self.i, self.keys = 0, list(datasets.keys())
        self.offset = 0

    # iterator that goes through each dataset in data
    # the samples list is a list of samples in the dataset
    # so if training has 10 samples and validation has 15
    # then validation starts at data.dataset[10]
    def __next__(self):
        if self.i == len(self.keys):
            raise StopIteration
        dataset_name = self.keys[self.i]
        dataset = self.datasets[dataset_name]

        n = len(dataset)
        assert len(dataset.samples) >= self.offset + n
        samples = dataset.samples[self.offset:self.offset + n]

        self.i += 1
        self.offset += n

        return dataset_name, samples

class FeatureCountProfile:
    # features at a label for that statistic and a function that takes
    # a list of samples and returns that statistic
    def __init__(self, samples, features, logdir):
        self.prof = {}
        for feature_name, get_feature in features:
            self.prof[feature_name] = FeatureCountProfile.get_counts(get_feature, samples)
        self.logdir = logdir
    
    def get_counts(extract_func, samples):
        counts = Counter()
        for sample in samples:
            counts.update(extract_func(sample))
        return counts

    def top_percents(self, name, top=5, digits=2):
        counts = self.prof[name]
        total = 0
        for k in counts:
            total += counts[k]
        
        percents = {}
        for k, v in counts.most_common(top):
            percents[k] = f"{round(100 * v / total, digits)}%"
        return percents

    def write_to_csv(self, features, fileprefix):
        for feature, _ in features:
            with open(os.path.join(self.logdir, f"{fileprefix}_{feature}.csv"), "w", newline="") as f:
                fieldnames = [feature, "count"]
                writer = csv.writer(f)
                writer.writerow(fieldnames)
                for key, val in self.prof[feature].items():
                    writer.writerow([key, val])   

    # passing features manually most places so it is forwards compatibile
    # if we don't care about some of the features later on
    # features is the original 
    # TODO: feel like this feature passing interface might be more confusing
    # TODO: maybe manage the list of features in the DatasetExplorer
    def plot_profile(self, features, filename):
        plt.clf()
        fig, axs = plt.subplots(1, len(features))
        fig.suptitle('Histograms of feature categories')
        for i, (feature, _) in enumerate(features):
            counts = self.prof[feature].values()
            axs[i].hist(counts, bins=10, density=True)
            axs[i].title.set_text(feature)
        plt.savefig(os.path.join(self.logdir, f'{filename}.png'))
    
    # feature is the string label
    # really for use by the DatasetExplorer
    # TODO: feel like this feature passing interface might be more confusing
    def label_list(self, feature):
        return list(self.prof[feature].keys())

    def get_feature(self, feature):
        return self.prof[feature]