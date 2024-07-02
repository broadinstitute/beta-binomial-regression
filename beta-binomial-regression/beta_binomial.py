import pandas as pd
import scanpy as sc
import numpy as np
import torch
import os
import re

from math import ceil

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_value_
import torch.optim as optm
from torch.autograd import Variable
from torch.autograd import grad

from sklearn.preprocessing import label_binarize
from pyro.distributions import GammaPoisson



###########################################################################
#  Likelihood Equation Functions
###########################################################################

def _log_beta(x, y):
    return torch.lgamma(x) + torch.lgamma(y) - torch.lgamma(x + y)

# Function for calculating the betabinomial log likelihood, with option for no total counts passed
def betabinomial_logprob(counts, alpha, beta, total_count=None):
    if total_count is None:
        total_count = counts.sum(axis=1, keepdims=True)

    return (_log_beta(counts + alpha, total_count - counts + beta) -
            _log_beta(alpha, beta))

# same log likelihood but summed to get loglikelihood on a per-gene basis
def betabinomial_logprob_pergene(counts, alpha, beta, total_count=None):
    if total_count is None:
        total_count = counts.sum(axis=1, keepdims=True)

    return (_log_beta(counts + alpha, total_count - counts + beta) -
            _log_beta(alpha, beta)).sum(axis=0)


def betabinomial_logprob_1(value, total_count, alpha, beta):
    return (_log_beta(value + alpha, total_count - value + beta) -
            _log_beta(alpha, beta)).mean()

###########################################################################
#  Beta Binomial Distribution Function
###########################################################################

def fit_beta_binom(cell_counts, bg_alphas=None, bg_betas=None, num_bg=0, maxiter=10, matrix=False):
    """ Fit Beta-Binomial Distribution to the Data

    Used to get distribution fits for background cells, that serve as our intiial guess in the regression
    Run on NC cells only, for example, to get baseline signal of each gene

    PARAMETERS
    -----------
        cell_counts: ANNDATA object or cell counts matrix pulled from anndata object

        bg_alphas, bg_betas: background beta distribution for fake cells we want to introduce

        num_bg: # fake cells to introduce

        matrix: defines cell_counts type, default false (cell_counts should be anndata)

    RETURNS
    -------
        alphas, betas: 1 x gene matrices with fitted alphas and betas. They serve as parameters for a beta distribution describing each gene

        initial_betas: 1 x gene matrix with initial betas guessed using Minka math
    """

    # extract counts matrix, and append fake BG cells
    if matrix:
        counts = cell_counts
    else:
        counts = cell_counts.X.A

    num_cells, num_genes = counts.shape
    # TODO: synchronize with version on terra
    # if totals is none ...
    totals = counts.sum(axis=1, keepdims=True)

    if not matrix:
        assert((totals > 10).all())

    if bg_alphas is None:
        assert num_bg == 0, "can't fake cells if also estimating bg rate"

    if num_bg > 0:
        # add background cells
        median_total = np.median(totals.ravel())
        bg_cell = median_total * bg_alphas / (bg_alphas + bg_betas)
        counts = np.vstack([counts] + [bg_cell] * num_bg)
        totals = counts.sum(axis=1, keepdims=True)

    # initial guess - moment matching - Minka 2000 (19)-(21)
    ratios = (counts / totals.astype(float))
    expected_alpha = ratios.mean(axis=0)
    expected_beta = (1 - ratios).mean(axis=0)
    expected_alpha_sq = (ratios ** 2).mean(axis=0)
    mult = (expected_alpha - expected_alpha_sq) / (expected_alpha_sq - expected_alpha ** 2)
    alphas = mult * expected_alpha
    betas = mult * expected_beta

    # available to save off for more info
    initial_betas = betas

    # move to torch
    alphas = torch.tensor(alphas).reshape((1, -1))
    betas = torch.tensor(betas).reshape((1, -1))
    totals = torch.tensor(totals).reshape((-1, 1))
    counts = torch.tensor(counts)

    for itr in range(maxiter):
        NLL = - betabinomial_logprob_1(counts, totals, alphas, betas)
        print(f"NegLogLikelihood {NLL.numpy():.4f}")

        alpha_old = alphas
        beta_old = betas

        # Minka equation 55
        numerator_a = (torch.digamma(counts + alpha_old) - torch.digamma(alpha_old)).sum(axis=0, keepdims=True)
        numerator_b = (torch.digamma(totals - counts + beta_old) - torch.digamma(beta_old)).sum(axis=0, keepdims=True)

        denominator = (torch.digamma(totals + alpha_old + beta_old) - torch.digamma(alpha_old + beta_old)).sum(axis=0, keepdims=True)

        # prevent extreme shifts by clipping
        alphas = alpha_old * (numerator_a / denominator).clip(1 / 2, 2)
        betas = beta_old * (numerator_b / denominator).clip(1 / 2, 2)

        print("   ", abs(numerator_a / denominator).max().numpy(), abs(numerator_b / denominator).max().numpy())


    return alphas, betas, initial_betas


###########################################################################
#  Beta Binomial Regression Functions
###########################################################################

def generate_tap_features(cell_counts, guide_map, delete=True, group=False):
    """ This function is capable of handling tap_seq data (higher MOI) for generating the features matrix used in BBR.
    See generate_features below.
    WARNING: Will need to be adjusted/changed depending on what your 'negative control' guides are called in the experiment.
    """
    num_cells, num_genes = cell_counts.shape
    print(num_cells, num_genes)
    guide_list = set(guide_map.values())
    if group:
        replacements = [(r'safe.*', "nonguide"), (r'negative.*',"nonguide")]
        for orig, repl in replacements:
            guide_list = np.unique(np.array(list(map(lambda v: re.sub(orig, repl, v) , guide_list))))

    feature_codes, uni_feature_names = pd.factorize(guide_list)

    binarized_features = np.empty(((num_cells), len(feature_codes)))

    ## can this step be sped up/it should be
    for i, guide_type in enumerate(uni_feature_names):
        if guide_type == 'nonguide':
            binarized_features[:, i] = (cell_counts.obs.target_names.str.contains('negative_control|safe_targeting', regex=True)).astype(int)
        else:
            binarized_features[:, i] = cell_counts.obs.target_names.str.contains(guide_type).astype(int)

    # Delete NC, nontarget, and safe from the features matrix
    if delete:
        if group:
            to_drop = ['nonguide']
        else:
            to_drop = ['negative_control', 'safe_targeting']
        drop_idx = np.where(np.isin(uni_feature_names, to_drop))[0]
        binarized_features = np.delete(binarized_features, drop_idx, axis=1)
        uni_feature_names = np.delete(uni_feature_names, drop_idx)

    # add in extra features "log(num_features)"
    binarized_features = np.column_stack((binarized_features, np.log(cell_counts.obs.num_features)))
    features = torch.tensor(binarized_features).double()
    uni_feature_names = np.hstack((uni_feature_names, ["log(num_features)"]))

    # don't normalize upon return
    return uni_feature_names, features


def normalize(features):
        return (features - features.mean(axis=0)) / features.std(axis=0)


def generate_features(cell_counts, cc=True, working_guides=True, permuted=False):
    """ Generate the feature matrix used in regression
    Called directly in the regression method

    PARAMETERS
    -----------

    cell_counts: An AnnData object with n_obs × n_vars (cells x genes)

    cc: A bool, indicating if features should include cell cycle

    working_guides: A bool, indicating if features should be indicated by working guides only

        e.g. SOX2-1 and SOX2-2 are working, but not SOX2-3. If not True, encode all cells that received any sox2 guide as 1

    permuted: A bool, indicating if features are called with permuted guide calls (for simulated KD)

    RETURNS
    -------

    Features order: A Categorical index variable, with codes (object 0 in the index) corresponding to the ordering of the guide columns in the feature matrix

        WARNING: This label list does not include S_score and G2M score, only guides. S_score and G2M score will always be last in features matrix if included.

    Features: A Cell x Feature Matrix

        First features will be the guides (as ordered in features order object).
        Last 2 features will be cell cycle features (S score, G2M score) if cc=True (default True)
        Which sub-guides (eg. SOX2-1, SOX2-2) are included depends on if you want working guides only (as defined by working_features column in anndata) or all guides other than control
        WARNING: this function is very specific to the perturb-seq data as of now. Do not use blindly without knowing what data looks like.
            For example, controls are only labeled 'NC' here, and we have a specific list of guides we deem as working
        All cell x guide portion of matrix should be 'one hot', meaning each cell has a 1 if it had that guide called else 0

    """


    if (working_guides):
        to_drop = 'No_working_guide'
        if (permuted):
            feature_codes, unique_features = pd.factorize(cell_counts.obs.perm_working_features)
        else:
            feature_codes, unique_features = pd.factorize(cell_counts.obs.working_features)
    else:
        to_drop = 'NC'
        if (permuted):
            feature_codes, unique_features = pd.factorize(cell_counts.obs.perm_feature_call)
            cell_threshold = cell_counts.obs.perm_feature_call.value_counts() > 50
        else:
            feature_codes, unique_features = pd.factorize(cell_counts.obs.feature_call)
            cell_threshold = cell_counts.obs.feature_call.value_counts() > 50

    features_1 = label_binarize(feature_codes, classes=np.unique(feature_codes))

    # don't include NC as a feature / non_working guides
    if not working_guides:
        features_1 = features_1[:, np.where(unique_features[cell_threshold])[0]]
        unique_features = unique_features[cell_threshold]

    features_1 = np.delete(features_1, np.where(unique_features == to_drop)[0][0], axis=1)
    if cc == True:
        features_2 = np.array(cell_counts.obs.S_score)
        features_3 = np.array(cell_counts.obs.G2M_score)
        features = torch.tensor(np.vstack((features_1.T, features_2, features_3)).T)
    else:
        features = torch.tensor(features_1).double()
    return unique_features[unique_features != to_drop], features

def generate_features_generic(counts, delete_names=None, column='feature_call', cc=True, filter_guides_thresh=None):
    """
    Generate features tensor.
    This function will work on both low moi and high moi data!

    INPUTS
    _______________________

    counts: anndata object for bbr

    delete_names: np.array, list of non-targeting guides to delete from feature matrix.
        If passing column='working_features', this can simply contain 'No_working_guide' or other uniform name for all control guides.

    column: Name of column in counts.obs to use for features. If high MOI, must be comma ',' separated guide names.

    cc: whether or not to include cell cycle features. only works if counts.obs has 'S_score' and 'G2M_score' columns.

    RETURNS
    _________________________

    Features order: Index variable, corresponding to the ordering of the guide columns in the feature matrix

        WARNING: This label list does not include S_score and G2M score, only guides. S_score and G2M score will always be last in features matrix if included.

    Features: A Cell x Feature Matrix

        First features will be the guides (as ordered in features order object).
        Last 2 features will be cell cycle features (S score, G2M score) if cc=True (default True)
        Which sub-guides (eg. SOX2-1, SOX2-2) are included depends on if you want working guides only (as defined by working_features column in anndata) or all guides other than control

        All cell x guide portion of matrix should be 'one hot', meaning each cell has a 1 if it had that guide called else 0


    """
    # this is magical, so much simpler, so much easier, thank you pd str split get dummies <3
    feature_df = counts.obs[column].str.get_dummies(',')
    if delete_names is not None:
        feature_df.drop(delete_names, axis=1, inplace=True)

    if filter_guides_thresh is not None:
        # pass a threshold for # cells a guide has to be in
        feature_df = feature_df.loc[:, feature_df.sum(axis=0) > filter_guides_thresh]

    uni_feature_names = feature_df.columns

    if cc:
        feature_df['S_score'] = counts.obs.S_score
        feature_df['G2M_score'] = counts.obs.G2M_score

    features = torch.tensor(feature_df.values).double()

    return uni_feature_names, features


def sgd_optimizer(cell_counts, a_NC, b_NC, maxiter=100, priorval=.075, lr=.001, subset=False, genelist=None, norm=False, weights=None, int_old=None, features=None, features_order=None, permuted=False, cc=True):
    """
    Beta-Binomial Regression for scRNA-seq Data.

    PARAMETERS
    ----------

    cell_counts: An AnnData object with n_obs × n_vars (cells x genes)

        At the very least, obs should contain 'feature_call', 'working_features', 'S_score', 'G2M_score' columns.
        Additionally, if running on a permuted data set, obs should contain 'perm_feature_call', 'perm_working_features'.

        If running on downsampled dataset, var should contain 'Downsampled' column, indicating if a gene has been artificially downsampled

        E.g. cell_counts:
            AnnData object with n_obs × n_vars = 6631 × 13577
                obs: 'n_genes_by_counts', 'total_counts', 'total_counts_mt', 'pct_counts_mt', 'total_counts_ribo', 'pct_counts_ribo', 'feature_call', 'working_features', 'n_genes', 'S_score', 'G2M_score', 'phase', 'perm_feature_call', 'perm_working_features'
                var: 'gene_ids', 'feature_types', 'genome', 'mt', 'ribo', 'n_cells_by_counts', 'mean_counts', 'pct_dropout_by_counts', 'total_counts', 'Downsampled'

    a_NC, b_NC: Two 1 x n_vars matrices containing background distributions for each gene

        Based on the alphas and betas fit to the negative control data in fit_beta_binom.

    maxiter: An int indicating number of iterations

    priorval: A float

            Assuming normal distribution, the priorval is our prior distribution sigma value to indicate prior on weights.
            This value can be obtained by running the regression with a weak prior (for your data, based on its standard deviation. For example, 1, 10),
            and getting the standard deviation of the weights from that regression.
            e.g. priorval = 0.1, 0.3, 0.075

    lr: the learning rate for the optimizer

    subset: bool, indicatig if the regression should be run on a specific subset of genes

    genelist: subset list of genes to run regression on. only passed if subset=True

    norm: bool, indicating if method should use normalized features. Default false, to keep features matrix 'one hot'

    weights, int_old, features, features_order: Torch.tensor objects corresponding to the weight, intercept, or features tensors in the regression.

        weights, int_old: These tensors are only passed if one does not want the tensors to be initialized to zero in the regression.

        features, features_order: These tensors are directly passed if we do not want the standard feature generation to be called.
            For example, a tap seq features matrix (still needs to be right size, format, etc.) can be passed after calling generate_tap_features() on the same cell_counts object passed to this function

    permuted: bool, indicating if the regression should be run using the permuted guide counts.

        This tool is specific and useful for the simulated KD runs, in which we permuted guide calls and knocked down a specific set of genes.
        Don't use otherwise, as regression will be run on meaningless guide calls.


    RETURNS
    _______

    w: A features x genes tensor.cpu object

        This contains the weights for each feature in each gene. Main regression output.
        Values are in ln scale but can be easily converted to log2 space, to serve as a log2 FC equivalent value

    new_mean: A cells x genes tensor.cpu object

        Mean values fit in the regression.

    new_s: A 1 x genes tensor.cpu object

        Scale values fit in the regression.

    delta_s: A 1 x genes tensor.cpu object

        Final adjustments made to scale values in the regression.

    loss_plt: A list of length maxiter

        Contains values of loss in each iteration, to be used for plotting to check for convergence

    intercept: A 1 x genes tensor.cpu object

        The intercept values in the regression

    features, features_order: same objects returned by generate_features

    second_derivative: A features x genes tensor.cpu object

        The second derivatives from the final optimization step. Used for generating p-values.


    """

    means = a_NC  / (a_NC + b_NC)
    s = a_NC + b_NC

    counts = cell_counts.X.A
    totals = counts.sum(axis=1, keepdims=True)
    print("totals: ", totals)

    if subset:
        # TODO: synchronize with version on terra, that allows for subsetting in the a_NC and b_NC objects prior to running this func.
        geneidx = [cell_counts.var.index.get_loc(item) for item in genelist]
        cell_counts = cell_counts[:, genelist]
        counts = cell_counts.X.A
        means = means[:, geneidx]
        s = s[:, geneidx]

    num_cells, num_genes = counts.shape

    if features is None:
        features_order, features = generate_features(cell_counts, cc=cc, working_guides=True, permuted=permuted)
    features = features.float()

    if norm:
        features = normalize(features)

    num_features = features.shape[1]

    # get initial tensors for regression
    w = torch.zeros([num_features, num_genes]).float().cuda()
    delta_s = torch.zeros((1, w.shape[1])).cuda()
    intercept = torch.zeros(delta_s.shape).cuda()
    means = torch.tensor(means).cuda()
    s = torch.tensor(s).cuda()
    counts = torch.tensor(counts).cuda()
    totals = torch.tensor(totals).cuda()

    if weights is not None:
        w = weights.clone().detach()

    if int_old is not None:
        intercept = int_old.clone().detach()

    w.requires_grad = True
    delta_s.requires_grad = True
    intercept.requires_grad = True

    print(w)

    # Use an SGD optimizer

    optimizer = optm.SGD([w, delta_s, intercept], lr=lr)
    loss_plt = []

    # run the optimization
    for i in range(maxiter):

        optimizer.zero_grad()

        new_mean = means * torch.exp(intercept + torch.matmul(features, w))
        new_mean = new_mean.float()
        new_s = s * torch.exp(delta_s)

        ### prior should get bigger as w gets bigger
        prior_w = priorw(w, priorval)
        ll = betabinomial_logprob(counts, new_mean * new_s, (1 - new_mean) * new_s, totals)
        loss = - (ll.sum() + prior_w.sum())
        print(loss, abs(w).max(), abs(delta_s).max())

        # get rid of retain_graph ?
        loss.backward(retain_graph=True)
        clip_grad_value_([w, delta_s, intercept], 3)
        optimizer.step()
        loss_plt.append(loss.cpu().detach().numpy())

    first_derivative = grad(loss, w, create_graph=True)[0]
    second_derivative = grad(first_derivative.sum(), w)[0]

    # [v.cpu().detach().numpy() for v in [w, new_mean, delta_s, ...]]
    return [v.cpu().detach().numpy() for v in [w, new_mean, new_s, delta_s, intercept, features, second_derivative]] + [loss_plt, features_order]


# get kde plots from w and get std

def priorw(w, sigma):
    return - (((w / sigma) ** 2) / 2)