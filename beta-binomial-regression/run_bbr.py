import pandas as pd
import scanpy as sc
import numpy as np
import torch
from scipy.stats import norm
from beta_binomial import fit_beta_binom, sgd_optimizer

def make_bbr_df(weights, counts, guide_order, cc=True, subset=False, genelist=None, log2_space=True, orig_counts=None):
    if log2_space:
        ratio = np.log(10) / np.log2(10)
        weights = weights / ratio # convert from ln to log2

    regr_scores = pd.DataFrame(index=genelist if subset else counts.var_names, data=weights.T,
                                columns=np.hstack((guide_order, ["S_score", "G2M_score"])) if cc else np.hstack((guide_order)))

    if subset:
        # need to make sure we scale to original counts
        normalizer = orig_counts.copy()
    else:
        normalizer = counts.copy()
    sc.pp.normalize_per_cell(normalizer, counts_per_cell_after=1e6)
    df = pd.DataFrame(normalizer.X.mean(axis=0).T, index=normalizer.var.index, columns=['mean_TPM'])
    regr_scores = regr_scores.merge(df, left_index=True, right_index=True)

    Day_compare = regr_scores.reset_index().rename(columns={"index":"gene"})
    Day_compare['mean_TPM_cat'] = pd.qcut(Day_compare.mean_TPM, q=10, precision=0)

    return Day_compare

def get_pvalues_df(second_deriv, w, features_order, genes):
    sigma = np.sqrt( 1 / second_deriv)
    zscore = w / sigma
    pvalues = (norm.sf(abs(zscore))*2)
    pvalues_df = pd.DataFrame(pvalues.T, columns=np.hstack((features_order, ["S_score", "G2M_score"])), index=genes)

    return pvalues_df


def run_whole_bbr(counts, group_name, split=False, priorval=.1, genelist=None, permuted=False, no_ds_counts=None):
    # group_name = 'D0' or 'D2'... etc.
    torch.set_default_tensor_type('torch.FloatTensor')

    if permuted:
        NC_counts = counts[counts.obs.perm_feature_call == 'NC']
    else:
        NC_counts = counts[counts.obs.feature_call == 'NC']
    a_NC, b_NC, _ = fit_beta_binom(NC_counts)

    torch.set_default_tensor_type('torch.cuda.FloatTensor')


    if split:
        # for split, we don't give it a passed genelist, instead we automatically split the genes into two (approximate halves for our data)
        regression_output = sgd_optimizer(counts, a_NC, b_NC, lr=0.001, maxiter=3000, priorval=priorval, subset=True, genelist=counts.var_names[0:5000], permuted=permuted)
        regression_output_2 = sgd_optimizer(counts, a_NC, b_NC, lr=0.001, maxiter=3000, priorval=priorval, subset=True, genelist=counts.var_names[5000:], permuted=permuted)

        weights_1, _, _, _, _, features_1, second_deriv_1, loss_plt_1, features_order = regression_output
        weights_2, _, _, _, _, features_2, second_deriv_2, loss_plt_2, features_order_2 = regression_output_2
        weights = np.hstack((weights_1, weights_2))
        second_deriv = np.hstack((second_deriv_1, second_deriv_2))

    # if we aren't splitting, we are either running it on the full dataset or passing a specific gene list to run on, with permutation specified as well
    # for example, genelist could be the downsampled genes only

    elif genelist is not None:
        regression_output = sgd_optimizer(counts, a_NC, b_NC, lr=0.001, maxiter=3000, priorval=priorval, subset=True, genelist=genelist, permuted=permuted)
        weights, _, _, _, _, features, second_deriv, loss_plt, features_order = regression_output
        subset_counts = counts.copy()[:, genelist]
        pvalues_df = get_pvalues_df(second_deriv, weights, features_order, genelist)
        bbr_df = make_bbr_df(weights, subset_counts, features_order, subset=True, genelist=genelist, orig_counts=counts if no_ds_counts is None else no_ds_counts)
        return bbr_df, pvalues_df

    else:
        regression_ouptut = sgd_optimizer(counts, a_NC, b_NC, lr=0.001, maxiter=3000, priorval=priorval, permuted=permute)
        weights, _, _, _, _, features, second_deriv, loss_plt, features_order = regression_output


    pvalues_df = get_pvalues_df(second_deriv, weights, features_order, counts.var.index)

    bbr_df = make_bbr_df(weights, counts, features_order)

    #bbr_df.to_csv(f'data/WTC_100TFs_10x_{group_name}_bbr_log2fc_results.csv')
    #pvalues_df.to_csv(f'data/WTC_100TFs_10x_{group_name}_bbr_pvalues_results.csv')

    return bbr_df, pvalues_df
