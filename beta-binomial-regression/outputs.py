import pandas as pd
import numpy as np
import torch
import scanpy as sc

def make_bbr_df(weights, counts, guide_order, cc=True, subset=False, genelist=None, log2_space=True):
    if log2_space:
        ratio = np.log(10) / np.log2(10)
        weights = weights / ratio # convert from ln to log2
    
    regr_scores = pd.DataFrame(index=genelist if subset else counts.var_names, data=weights.T.detach().numpy(), 
                                columns=np.hstack((guide_order, ["S_score", "G2M_score"])) if cc else np.hstack((guide_order)))
    
    normalizer = counts.copy()
    sc.pp.normalize_per_cell(normalizer, counts_per_cell_after=1e6)
    df = pd.DataFrame(normalizer.X.mean(axis=0).T, index=normalizer.var.index, columns=['mean_TPM'])
    regr_scores = regr_scores.merge(df, left_index=True, right_index=True)
    
    Day_compare = regr_scores.reset_index().rename(columns={"index":"gene"})
    Day_compare['mean_TPM_cat'] = pd.qcut(Day_compare.mean_TPM, q=10, precision=0)

    return Day_compare


def read_MAST(guide, file_str, ds_amount='50'):
    return pd.read_csv(f'D0_perm_ds_{ds_amount}/MAST_{guide}_{file_str}.txt', sep='\t')

## this needs to be updated and changed
def make_comparison_df(weights, counts, orig_counts, guide_order, day, mast_guide_list, file_str, cc=True, subset=False, genelist=None, log2_space=True, ds_amount='50'):
    # e.g counts = D2_counts_NC_SOX2
    if log2_space:
        ratio = np.log(10) / np.log2(10)
        weights = weights / ratio # convert from ln to log2
    if cc == True:
        if subset:
            regr_scores = pd.DataFrame(index=genelist, data=weights.T.detach().numpy(), columns=np.hstack((guide_order,
                                                                                                         ["S_score", "G2M_score"])))
        else:
            regr_scores = pd.DataFrame(index=counts.var_names, data=weights.T, columns=np.hstack((guide_order,
                                                                                                         ["S_score", "G2M_score"])))
    else:
        if subset:
            regr_scores = pd.DataFrame(index=genelist, data=weights.T.detach().numpy(), columns=np.hstack((guide_order)))
        else:
            regr_scores = pd.DataFrame(index=counts.var_names, data=weights.T, columns=np.hstack((guide_order)))

    normalizer_og = orig_counts.copy()
    sc.pp.normalize_per_cell(normalizer_og, counts_per_cell_after=1e6)
    df = pd.DataFrame(normalizer_og.X.mean(axis=0).T, index=normalizer_og.var.index, columns=['mean_TPM'])
    regr_scores = regr_scores.merge(df, left_index=True, right_index=True)

    normalizer = counts.copy()
    sc.pp.normalize_per_cell(normalizer, counts_per_cell_after=1e6)

    permuted_df_TPMS = pd.DataFrame({'NC mean TPM' : normalizer[normalizer.obs.perm_feature_call == 'NC', :].X.mean(axis=0).A.reshape(-1)},
                               index = normalizer.var.index)
    for TF in "HHEX HNF1B SOX2 YBX1 ID1".split():
        permuted_df_TPMS[f'{TF} mean TPM'] = normalizer[normalizer.obs.perm_working_features == TF, :].X.mean(axis=0).A.reshape(-1)

    Day_compare = regr_scores.reset_index().rename(columns={"index":"gene"})
    Day_compare = Day_compare.merge(permuted_df_TPMS.reset_index().rename(columns={"index":"gene"}))

    for mast_guide in mast_guide_list:
        df = read_MAST(mast_guide, file_str, ds_amount)
        logfc = df[["gene", "logFC"]].rename(columns={'logFC':f'{mast_guide} MAST logFC'})
        Day_compare = logfc.merge(Day_compare, on='gene')

    Day_compare['mean_TPM_cat'] = pd.qcut(Day_compare.mean_TPM, q=10, precision=0)

    return Day_compare
