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


def run_whole_bbr(
    counts,
    split=False,
    priorval=0.1,
    genelist=None,
    permuted=False,
    no_ds_counts=None,
    cc=True,
    maxiter=3000,
):
    """
    Example run of an entire BBR regression, given filtered counts.
    This function was specifically developed for our low-moi 8TF perturb experiment.
    The counts anndata object has features assigned based on guides for each TF in the 'working_features' column,
        as either a guide name or 'No_working_guide'. We pass this directly to the optimizer for generating the features matrix.
    There is also code built in for running this on the permuted version of this data, with all
        data in corresponding 'perm_' columns.

    Users can use this as an example for running BBR and getting the output dataframes.

    Because genes are indpendent in BBR, users can split the genes used in runs in different ways and still
        calculate the same result for each gene-guide pair.

    Genelist option:
        Pass a specific genelist and BBR will only be run on those genes. Best way to save time.

    Split option:
        BBR regression with 5000+ genes takes a lot of computing power and cuda memory.
        If you are not running on a subset of targeted genes only (passing a genelist), the split option will split
        your data into the first 5000 genes and the rest, as to not overload the memory.
        (not the cleanest method)

        Users are also advised to clear their cuda memory between runs if necessary.
    """

    torch.set_default_tensor_type("torch.FloatTensor")

    if permuted:
        NC_counts = counts[counts.obs.perm_working_features == "No_working_guide"]
    else:
        NC_counts = counts[counts.obs.working_features == "No_working_guide"]
    a_NC, b_NC, _ = fit_beta_binom(NC_counts)

    torch.set_default_tensor_type("torch.cuda.FloatTensor")

    if split:

        # for split, we don't give it a passed genelist, instead we automatically split the genes into two (approximate halves for our data)
        regression_output = sgd_optimizer(
            counts,
            a_NC,
            b_NC,
            lr=0.001,
            maxiter=maxiter,
            priorval=priorval,
            subset=True,
            genelist=counts.var_names[0:5000],
            features_column='perm_working_features' if permuted else 'working_features',
            delete_names=['No_working_guide'],
            cc=cc,
        )
        regression_output_2 = sgd_optimizer(
            counts,
            a_NC,
            b_NC,
            lr=0.001,
            maxiter=maxiter,
            priorval=priorval,
            subset=True,
            genelist=counts.var_names[5000:],
            features_column='perm_working_features' if permuted else 'working_features',
            delete_names=['No_working_guide'],
            cc=cc,
        )

        (
            weights_1,
            _,
            _,
            _,
            _,
            features_1,
            second_deriv_1,
            loss_plt_1,
            features_order,
        ) = regression_output

        (
            weights_2,
            _,
            _,
            _,
            _,
            features_2,
            second_deriv_2,
            loss_plt_2,
            features_order_2,
        ) = regression_output_2

        weights = np.hstack((weights_1, weights_2))
        second_deriv = np.hstack((second_deriv_1, second_deriv_2))
        loss_plt = [loss_plt_1, loss_plt_2]

    # if we aren't splitting, we are either running it on the full dataset or passing a specific gene list to run on,
    #   with permutation specified as well
    # for example, genelist could be the downsampled genes only

    elif genelist is not None:
        regression_output = sgd_optimizer(
            counts,
            a_NC,
            b_NC,
            lr=0.001,
            maxiter=maxiter,
            priorval=priorval,
            subset=True,
            genelist=genelist,
            features_column='perm_working_features' if permuted else 'working_features',
            delete_names=['No_working_guide'],
            cc=cc,
        )
        weights, _, _, _, _, features, second_deriv, loss_plt, features_order = (
            regression_output
        )
        subset_counts = counts.copy()[:, genelist]
        pvalues_df = get_pvalues_df(
            second_deriv, weights, features_order, genelist, cc=cc
        )
        bbr_df = make_bbr_df(
            weights,
            subset_counts,
            features_order,
            cc=cc,
            subset=True,
            genelist=genelist,
            orig_counts=counts if no_ds_counts is None else no_ds_counts,
        )
        return bbr_df, pvalues_df

    else:
        # run regression on all genes.
        regression_ouptut = sgd_optimizer(
            counts,
            a_NC,
            b_NC,
            lr=0.001,
            maxiter=maxiter,
            priorval=priorval,
            features_column='perm_working_features' if permuted else 'working_features',
            delete_names=['No_working_guide'],
            cc=cc,
        )
        weights, _, _, _, _, features, second_deriv, loss_plt, features_order = (
            regression_output
        )

    pvalues_df = get_pvalues_df(
        second_deriv, weights, features_order, counts.var.index, cc=cc
    )

    # TODO: Potentially, change things around here, so users can pass subset counts (no_ds_counts changed to just alt_counts or something)
    # and we just pass to this function similarly to how we do above
    # might be no reason to not just add this now, but idk I'll leave it as is
    bbr_df = make_bbr_df(weights, counts, features_order, cc=cc)

    return bbr_df, pvalues_df, loss_plt
