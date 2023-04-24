import pandas as pd
import scanpy as sc
import numpy as np


pd.options.mode.chained_assignment = None  # default='warn'

## this should change depending on user input
work_list = "TBXT_1,TBXT_2,TBXT_3,TBXT_10,HAND1_8,HAND1_2,HAND1_10,HAND1_5,HAND1_3,HAND1_7,HAND1_4,HAND1_1,SOX2_1543,SOX2_1553,YBX1_823,YBX1_821,ID1_9,ID1_4,ID1_10,ID1_8,ID1_5,ID1_3,ID1_1,ID1_7,GATA3_9,GATA3_2,GATA3_1,GATA3_3,GATA3_4,GATA3_5,GATA3_7,HHEX_4,HHEX_7,HHEX_1,HHEX_6,HHEX_2,HHEX_9,HHEX_10,HHEX_3,HNF1B_1,HNF1B_3,HNF1B_6,HNF1B_10,HNF1B_5,HNF1B_2,HNF1B_7,HNF1B_8,HNF1B_4"

def get_cell_cycles(adata):
    cell_cycle_genes = [x.strip() for x in open('data/regev_lab_cell_cycle_genes.txt')]
    s_genes = cell_cycle_genes[:43]
    g2m_genes = cell_cycle_genes[43:]
    cell_cycle_genes = [x for x in cell_cycle_genes if x in adata.var_names]
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)

def filter_guide_counts(counts, guide, alphas, betas):
    # to run after filter_all_counts if want genes and cells for a specific guide from that day!

    guide_barcodes = counts.obs.feature_call.str.contains(guide)
    NC_barcodes =  counts.obs.feature_call.str.contains("NC")

    counts_NC_guide = counts[list(NC_barcodes + guide_barcodes)].copy()
    counts_NC = counts[NC_barcodes].copy()
    print(counts_NC.shape)
    print(counts_NC_guide.shape)

    genes = list(set(counts_NC_guide.var_names) & set(counts_NC.var_names))
    counts = counts[:, genes].copy()
    counts_NC = counts_NC[:, genes].copy()
    counts_NC_guide = counts_NC_guide[:, genes].copy()

    print(counts_NC.shape)
    print(counts_NC_guide.shape)

    guide_alphas = alphas.loc[guide_barcodes, :]
    guide_betas = betas.loc[guide_barcodes, :]
    nc_alphas = alphas.loc[NC_barcodes, :]
    nc_betas = betas.loc[NC_barcodes, :]

    return(counts, counts_NC, counts_NC_guide, guide_alphas, guide_betas, nc_alphas, nc_betas)

def filter_all_counts(day_counts, day_CRISPR_counts, work_list=work_list):
    counts = day_counts.copy()
    CRISPR_counts = day_CRISPR_counts.copy()

    # filter mitochondria and ribosomes
    counts.var_names_make_unique()
    counts.var['mt'] = counts.var_names.str.startswith('MT-')
    counts.var['ribo'] =  counts.var_names.str.startswith(("RPS","RPL"))
    sc.pp.calculate_qc_metrics(counts, qc_vars=['mt','ribo'], percent_top=None, log1p=False, inplace=True)
    counts = counts[counts.obs['pct_counts_mt'] < 20, :]
    counts = counts[counts.obs['pct_counts_ribo'] > 10, :]
    guide_data = counts.obs.merge(CRISPR_counts[CRISPR_counts.cell_barcode.isin(counts.obs.index)][['cell_barcode', 'feature_call']].set_index('cell_barcode'),
                              left_index=True, right_index=True).copy()
    guide_data.feature_call = guide_data.feature_call.str.replace("-", "_")
    working_cells = guide_data.feature_call.isin(work_list.split(','))
    guide_data["working_features"] = np.nan
    guide_data.working_features[working_cells] = guide_data.feature_call[working_cells].str.split('_', expand=True)[0]
    guide_data.working_features[~working_cells] = 'No_working_guide'
    counts = counts[guide_data.index, :]
    counts.obs = guide_data

    sc.pp.filter_cells(counts, min_genes=200)
    cc_counts = counts.copy()
    get_cell_cycles(cc_counts)
    counts.obs[["S_score", "G2M_score", "phase"]] = cc_counts.obs[["S_score", "G2M_score", "phase"]]

    NC_barcodes = list(set(CRISPR_counts.cell_barcode[CRISPR_counts.feature_call.str.contains("NC")]) & set(counts.obs_names))
    counts.obs.replace(to_replace = r"NC.*", value="NC", regex=True, inplace=True)
    counts_NC = counts[NC_barcodes].copy()
    sc.pp.filter_genes(counts_NC, min_cells = counts_NC.shape[0]//20)

    genes = list(set(counts_NC.var_names))
    counts = counts[:, genes].copy()
    counts_NC = counts_NC[:, genes].copy()

    return counts, counts_NC


##### helpping for generalizing later, not finished!!!!!!!
def filter_try_all_counts(day_counts, day_CRISPR_counts, work_list, min_genes):
    counts = day_counts.copy()
    CRISPR_counts = day_CRISPR_counts.copy()

    counts.var_names_make_unique()
    guide_data = counts.obs.merge(CRISPR_counts[CRISPR_counts.cell_barcode.isin(counts.obs.index)][['cell_barcode', 'feature_call']].set_index('cell_barcode'),
                              left_index=True, right_index=True).copy()
    guide_data.feature_call = guide_data.feature_call.str.replace("-", "_")
    working_cells = guide_data.feature_call.isin(work_list.split(','))
    guide_data["working_features"] = np.nan
    guide_data.working_features[working_cells] = guide_data.feature_call[working_cells].str.split('_', expand=True)[0]
    guide_data.working_features[~working_cells] = 'No_working_guide'
    counts = counts[guide_data.index, :]
    counts.obs = guide_data

    sc.pp.filter_cells(counts, min_genes=min_genes)
    print(counts)
    cc_counts = counts.copy()
    
    cc = get_cell_cycles(cc_counts)
    if cc:
        counts.obs[["S_score", "G2M_score", "phase"]] = cc_counts.obs[["S_score", "G2M_score", "phase"]]
    
    if CRISPR_counts.feature_call.str.contains("NC").any():
        NC_barcodes = list(set(CRISPR_counts.cell_barcode[CRISPR_counts.feature_call.str.contains("NC")]) & set(counts.obs_names))
        counts.obs.replace(to_replace = r"NC.*", value="NC", regex=True, inplace=True)
        counts_NC = counts[NC_barcodes].copy()
        sc.pp.filter_genes(counts_NC, min_cells = counts_NC.shape[0]//20)
    else:
        NC_barcodes = list(set(counts.obs[counts.obs.working_features == 'No_working_guide'].index))
        counts_NC = counts[NC_barcodes].copy()
        sc.pp.filter_genes(counts_NC, min_cells = counts_NC.shape[0]//20)

    print(counts_NC)
    genes = list(set(counts_NC.var_names))
    counts = counts[:, genes].copy()

    return counts, counts_NC


def get_cell_cycles(adata):
    cell_cycle_genes = [x.strip() for x in open('regev_lab_cell_cycle_genes.txt')]
    s_genes = cell_cycle_genes[:43]
    g2m_genes = cell_cycle_genes[43:]
    cell_cycle_genes = [x for x in cell_cycle_genes if x in adata.var_names]
    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    if not set(cell_cycle_genes) & set(s_genes) or not set(cell_cycle_genes) & set(g2m_genes):
        return False
    sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
    
    return True
