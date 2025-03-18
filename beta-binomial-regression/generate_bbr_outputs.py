import pandas as pd
import scanpy as sc
import numpy as np
import torch
from scipy.stats import norm
from beta_binomial import fit_beta_binom, sgd_optimizer


def make_bbr_df(
    weights,
    counts,
    guide_order,
    cc=True,
    subset=False,
    genelist=None,
    log2_space=True,
    orig_counts=None,
):
    if log2_space:
        ratio = np.log(10) / np.log2(10)
        weights = weights / ratio  # convert from ln to log2

    regr_scores = pd.DataFrame(
        index=genelist if subset else counts.var_names,
        data=weights.T,
        columns=np.hstack((guide_order, ["S_score", "G2M_score"])) if cc else np.hstack((guide_order)),
    )

    if subset:
        # need to make sure we scale to original counts
        normalizer = orig_counts.copy()
    else:
        normalizer = counts.copy()

    sc.pp.normalize_per_cell(normalizer, counts_per_cell_after=1e6)
    df = pd.DataFrame(
        normalizer.X.mean(axis=0).T, index=normalizer.var.index, columns=["mean_TPM"]
    )
    regr_scores = regr_scores.merge(df, left_index=True, right_index=True)

    Day_compare = regr_scores.reset_index().rename(columns={"index": "gene"})

    return Day_compare


def get_pvalues_df(second_deriv, w, features_order, genes, cc=True):
    sigma = np.sqrt(1 / second_deriv)
    zscore = w / sigma
    pvalues = norm.sf(abs(zscore)) * 2
    pvalues_df = pd.DataFrame(
        pvalues.T,
        columns=np.hstack((features_order, ["S_score", "G2M_score"])) if cc else features_order,
        index=genes,
    )

    return pvalues_df


def adjust_weights_non_targeting(weights, row_id=-1):
    """
    Function for adjusting weights to non-targeting weight estimates.
        BBR relies on an initial guess for background cell distributions based on non-targeting cells (a_NC, b_NC).
        In certain cases, the initial guess may not be accurate.
        Passing "non-targeting" as a features column to the model allows for users to estimate adjustments for non-targeting guides per gene.
        Users can then subtract these non-targeting weights from all other guide-gene weights (to act as an intercept adjusment) for more accurate weights and p-values.

        Function should be called before generating the pvalue and bbr dataframes.

        Default assumes that "non-targeting" is the last column in the features matrix (row_id = -1)
    """
    # ADJUST THE WEIGHTS BY SUBTRACTING THE NEW NON-TARGETING WEIGHTS
    weights_adj = weights - weights[None, row_id, :]

    return weights_adj
