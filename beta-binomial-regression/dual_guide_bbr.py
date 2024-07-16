import pandas as pd
import scanpy as sc
import torch
import argparse
from beta_binomial import generate_features_generic, fit_beta_binom, sgd_optimizer
from run_bbr import make_bbr_df, get_pvalues_df


def filter_counts(counts, min_genes=200, min_cells=20):
    counts.var_names_make_unique()
    counts.var["mt"] = counts.var_names.str.startswith("MT-")
    counts.var["ribo"] = counts.var_names.str.startswith(("RPS", "RPL"))
    sc.pp.calculate_qc_metrics(
        counts, qc_vars=["mt", "ribo"], percent_top=None, log1p=False, inplace=True
    )
    counts = counts[counts.obs["pct_counts_mt"] < 20, :]
    counts = counts[counts.obs["pct_counts_ribo"] > 10, :]
    # delete mt or ribo genes
    counts = counts[:, counts.var.query("not (mt or ribo)").index]

    # Do filtering steps by num cells and genes
    sc.pp.filter_cells(counts, min_genes=min_genes)
    sc.pp.filter_genes(counts, min_cells=min_cells)

    return counts


def matches(s):
    a, b = s.split("|")
    return a.replace("dgA", "dgB") == b


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", dest="counts_h5", help="10x h5 filtered counts", type=str, required=True
    )
    parser.add_argument(
        "-f",
        dest="feature_calls",
        help="protospacer_calls_per_cell csv file, with cells and their feature calls",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    cts = sc.read_10x_h5(args.counts_h5)
    crispr = pd.read_csv(args.feature_calls)

    print("Filtering to dual guide data - cells with exactly 2 guides, that match in guide target.")
    crispr_keep = crispr.query("num_features == 2").copy()
    crispr_keep = crispr_keep[crispr_keep.feature_call.map(matches)].copy()

    print("Removing weird naming scheme (relevant to our data only), 'pos' and 'p1, p2'.")
    pattern = "|".join(["_pos", "_p1", "_p2"])
    crispr_keep["gene"] = (crispr_keep.feature_call.str.replace(pattern, "",
                                                                regex=True).str.split("_").str[-1])
    crispr_keep["feature_call"] = crispr_keep.feature_call.str.replace("|", ",")

    print("Merging cell feature calls with count matrix.")
    obs_data = cts.obs.merge(crispr_keep.set_index("cell_barcode"), left_index=True, right_index=True)
    cts_keep = cts[obs_data.index, :]
    cts_keep.obs = obs_data

    print(
        "Basic filtering of counts for mito, ribo, min number of expressed genes/cells. \n Min genes=200, Min cells=20."
    )
    cts_keep_filtered = filter_counts(cts_keep.copy())

    fg_keep = crispr_keep[~(crispr_keep.feature_call.str.startswith("dgA_NC") | crispr_keep.feature_call.str.startswith("dgA_J-NC"))]

    target_genes = fg_keep.gene.unique()
    cts_keep_filtered.obs["target"] = [
        x if "NC" not in x else "NC" for x in cts_keep_filtered.obs.gene
    ]

    print("Generate features matrix.")
    # cc always false here
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    # do not delete NC names unless specifying
    feature_names, features, feature_df = generate_features_generic(
        cts_keep_filtered, delete_names=None, column="target", cc=False
    )

    torch.set_default_tensor_type("torch.FloatTensor")
    tmp_nc = cts_keep_filtered[cts_keep_filtered.obs.target == "NC"]

    print("Drop any genes that have 0 total counts in NC")
    cts_keep_filtered = cts_keep_filtered[:, tmp_nc.X.toarray().sum(axis=0) > 0]
    print("Dropped ", (tmp_nc.X.toarray().sum(axis=0) == 0).sum(), " genes.")
    NC_counts = cts_keep_filtered[cts_keep_filtered.obs.target == "NC"]
    print(NC_counts.shape)

    print("Get background beta distributions fit to NC data.")
    a_NC, b_NC, _ = fit_beta_binom(NC_counts.X.toarray(), matrix=True)

    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    genelist = target_genes

    print("Running regression.")
    # pass in features we made manually
    regression_output = sgd_optimizer(
        cts_keep_filtered,
        a_NC,
        b_NC,
        lr=0.001,
        features=features,
        features_order=feature_names,
        maxiter=1000,
        priorval=1,
        subset=True,
        genelist=genelist,
        permuted=False,
        cc=False,
        sparse=True,
    )

    weights, _, _, _, _, features, second_deriv, loss_plt, features_order = (
        regression_output
    )
    subset_counts = cts_keep_filtered.copy()[:, genelist]
    pvalues_df = get_pvalues_df(
        second_deriv, weights, features_order, genelist, cc=False
    )
    bbr_df = make_bbr_df(
        weights,
        subset_counts,
        features_order,
        cc=False,
        subset=True,
        genelist=genelist,
        orig_counts=cts_keep_filtered,
    )

    return bbr_df, pvalues_df


if __name__ == "__main__":
    main()
