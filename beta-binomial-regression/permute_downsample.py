import pandas as pd
import numpy as np
import scanpy as sc
import anndata as ad



def get_downsampled_counts(day_counts, ds):

    downsampled = day_counts.copy()
    print(downsampled.X.data)
    sampling = np.random.binomial(day_counts.X.data.astype(int), ds)
    downsampled.X.data = sampling
    downsampled_counts = downsampled
    print(downsampled_counts.X.data)
    print(day_counts.X.A == downsampled_counts.X.A)

    return downsampled_counts

def permute_and_downsample(counts, keep, genelist=None):
    counts_perm = counts.copy()
    counts_perm.obs['perm_feature_call'] = counts_perm.obs.feature_call.sample(frac=1).values

    work_list = "TBXT_1,TBXT_2,TBXT_3,TBXT_10,HAND1_8,HAND1_2,HAND1_10,HAND1_5,HAND1_3,HAND1_7,HAND1_4,HAND1_1,SOX2_1543,SOX2_1553,YBX1_823,YBX1_821,ID1_9,ID1_4,ID1_10,ID1_8,ID1_5,ID1_3,ID1_1,ID1_7,GATA3_9,GATA3_2,GATA3_1,GATA3_3,GATA3_4,GATA3_5,GATA3_7,HHEX_4,HHEX_7,HHEX_1,HHEX_6,HHEX_2,HHEX_9,HHEX_10,HHEX_3,HNF1B_1,HNF1B_3,HNF1B_6,HNF1B_10,HNF1B_5,HNF1B_2,HNF1B_7,HNF1B_8,HNF1B_4"
    working_cells = counts_perm.obs.perm_feature_call.isin(work_list.split(','))
    counts_perm.obs["perm_working_features"] = np.nan
    counts_perm.obs.perm_working_features[working_cells] = counts_perm.obs.perm_feature_call[working_cells].str.split('_',expand=True)[0]
    counts_perm.obs.perm_working_features[~working_cells] = 'No_working_guide'

    # Downsample
    # first, get anndata object of all the counts that aren't from 'NC' in permuted version
    counts_perm_nonc = counts_perm[counts_perm.obs.perm_working_features != 'No_working_guide']

    # next, get 20-21 random genes to downsample within those cells
    if genelist is None:
        genesampling = counts_perm_nonc[:, counts_perm_nonc.var.sample(n = 18).index]
        if np.sum(genesampling.var.total_counts > 80000) <= 1:
            high_count_genes = counts_perm_nonc[:, counts_perm_nonc[:, counts_perm_nonc.var.total_counts > 80000].var.sample(n=2).index]
            genesampling = ad.concat((genesampling, high_count_genes), axis=1, merge='same')

    # or, pass "genelist", a list of genes you know you want to downsample
    else:
        genesampling = counts_perm_nonc[:, genelist]

    gene_indicator = np.in1d(counts_perm_nonc.var_names, genesampling.var_names)

    # downsample those counts
    downsampled_perm = ad.concat((counts_perm_nonc[:, ~gene_indicator], get_downsampled_counts(genesampling, keep)), axis=1, merge='same')
    # concat those counts with the 'NC & no working guide' cells
    counts_perm_ds = ad.concat((downsampled_perm, counts_perm[counts_perm.obs.perm_working_features == 'No_working_guide']), merge='same')
    counts_perm_ds.var['Downsampled'] = [gene in genesampling.var.index for gene in counts_perm_ds.var.index]

    return counts_perm_ds, counts_perm_nonc, genesampling

