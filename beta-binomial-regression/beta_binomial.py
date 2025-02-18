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


###########################################################################
#  Beta Binomial Distribution Function - Background Distributions
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
        NLL = - betabinomial_logprob(counts, alphas, betas, totals).nanmean()
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

def normalize(features):
        return (features - features.mean(axis=0)) / features.std(axis=0)


def generate_features_generic(counts, delete_names=None, column='feature_call', cc=True, filter_guides_thresh=None, str_split=','):
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

    filter_guide_thresh: Minimum number of cells a guide must be in to be used in the regression. Optional input.
        (ex. a guide must be in at least 100 cells for sufficient power)

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
    feature_df = counts.obs[column].str.get_dummies(str_split)
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

    return uni_feature_names, features, feature_df


def sgd_optimizer(cell_counts, a_NC, b_NC, maxiter=100, priorval=.075,
                  lr=.001, subset=False, genelist=None, norm=False,
                  weights=None, int_old=None, features=None,
                  features_order=None, cc=True,
                  numpy=False, sparse=False,
                  features_column='feature_call', delete_names=None):
    """
    Beta-Binomial Regression for scRNA-seq Data.

    PARAMETERS
    ----------

    cell_counts: An AnnData object with n_obs × n_vars (cells x genes)

        At the very least, obs should contain 'feature_call', column.
            generate_features_generic assumes features are in 'feature_call'

        Additionally, if running on a permuted data set, obs must contain 'perm_feature_call'
            and generate_features_generic assumes features are in 'perm_feature_call'

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

    numpy, sparse: parameters for handling the .X (actual raw counts matrix) data in the counts anndata object.
        TODO: deal with scanpy 10x functionality.
        Scanpy reads 10x scRNA-seq outputs differently depending on version.
        These parameters allow you to cast your anndata X object to a numpy array, whether sparse or already in array format.

    weights, int_old: Torch.tensor objects corresponding to the weight, intercept, or features tensors in the regression.

        weights, int_old: These tensors are only passed if one does not want the tensors to be initialized to zero in the regression.

    features, features_order: Torch.tensor objects corresponding to the features tensors in the regression.
        These tensors are directly passed if we do not want the standard feature generation to be called.
            * For example, user might want to specify min number of cells a guide should be in using the generate_features_generic function

    features_column, delete_names: Specific inputs to be passed to generate_features_generic for generating the features.
        Default values are the same as in generate_features_generic.
        If a user is running the model on permuted data, user can pass the scrambled guide calls column here.


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

    features, features_order: same objects returned by generate_features_generic

    second_derivative: A features x genes tensor.cpu object

        The second derivatives from the final optimization step. Used for generating p-values.


    """

    means = a_NC  / (a_NC + b_NC)
    s = a_NC + b_NC

    if sparse:
            counts = cell_counts.X.toarray()
    elif numpy:
        counts = cell_counts.X
    else:
        # X is not already in numpy object format
        counts = cell_counts.X.A

    # must calculate totals before doing any subsetting
    totals = counts.sum(axis=1, keepdims=True)
    print("totals: ", totals)

    if subset:
        # allows for
        if cell_counts.n_vars == means.shape[1]:
            geneidx = [cell_counts.var.index.get_loc(item) for item in genelist]
            means = means[:, geneidx]
            s = s[:, geneidx]
        cell_counts = cell_counts[:, genelist]

        # repeat code, not the cleanest, but have to deal with scanpy differences reading 10x
        if sparse:
            counts = cell_counts.X.toarray()
        elif numpy:
            counts = cell_counts.X
        else:
            # X is not already in numpy object format
            counts = cell_counts.X.A

        assert cell_counts.n_vars == means.shape[1]

    num_cells, num_genes = counts.shape

    if features is None:
        # assumes data is in 'feature_call' column, separated by commas. If permuted data, can pass 'perm_feature_call'
        features_order, features, _ = generate_features_generic(cell_counts, cc=cc, column=features_column, delete_names=delete_names)

    features = features.to(torch.float32).cuda()

    if norm:
        features = normalize(features)

    num_features = features.shape[1]

    # get initial tensors for regression
    # deliberately change size to int32 and float32 instead of larger types
    w = torch.zeros([num_features, num_genes]).float().cuda()
    delta_s = torch.zeros((1, w.shape[1])).cuda()
    intercept = torch.zeros(delta_s.shape).cuda()
    means = torch.tensor(means).float().cuda()
    s = torch.tensor(s).float().cuda()
    counts = torch.tensor(counts).int().cuda()
    totals = torch.tensor(totals).int().cuda()

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