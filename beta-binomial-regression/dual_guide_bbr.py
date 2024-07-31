import pandas as pd
import scanpy as sc
import torch
import argparse
from count_filtering import filter_counts
from beta_binomial import generate_features_generic, fit_beta_binom, sgd_optimizer
from run_bbr import make_bbr_df, get_pvalues_df


def matches(s):
    a, b = s.split("|")
    return a.replace("dgA", "dgB") == b


def main():
    """
    Entire run through for taking dual-guide 10x outputs and running BBR.
    Specific to dual guide experiment of 2024 (guide names not generic.)
    """
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

    # Read in your data
    cts = sc.read_10x_h5(args.counts_h5)
    crispr = pd.read_csv(args.feature_calls)

    # Deal with dual guide set-up
    print("Filtering to dual guide data - cells with exactly 2 guides, that match in guide target.")
    crispr_keep = crispr.query("num_features == 2").copy()
    crispr_keep = crispr_keep[crispr_keep.feature_call.map(matches)].copy()

    print("Removing weird naming scheme (relevant to our data only), 'pos' and 'p1, p2'.")
    pattern = "|".join(["_pos", "_p1", "_p2"])
    crispr_keep["gene"] = (crispr_keep.feature_call.str.replace(pattern, "",
                                                                regex=True).str.split("_").str[-1])
    crispr_keep["feature_call"] = crispr_keep.feature_call.str.replace("|", ",")

    # Assign features to cells in object
    print("Merging cell feature calls with count matrix.")
    obs_data = cts.obs.merge(crispr_keep.set_index("cell_barcode"), left_index=True, right_index=True)
    cts_keep = cts[obs_data.index, :]
    cts_keep.obs = obs_data

    # Generic filtering
    print(
        "Basic filtering of counts for mito, ribo, min number of expressed genes/cells. \n Min genes=200, Min cells=20."
    )
    cts_keep_filtered = filter_counts(cts_keep.copy())

    fg_keep = crispr_keep[~(crispr_keep.feature_call.str.startswith("dgA_NC") | crispr_keep.feature_call.str.startswith("dgA_J-NC"))]

    target_genes = fg_keep.gene.unique()
    cts_keep_filtered.obs["target"] = [
        x if "NC" not in x else "NC" for x in cts_keep_filtered.obs.gene
    ]

    # Generate the features (optional but helpful)
    print("Generate features matrix.")
    # cc always false here
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    # do not delete NC names unless specifying
    feature_names, features, feature_df = generate_features_generic(
        cts_keep_filtered, delete_names=None, column="target", cc=False
    )

    torch.set_default_tensor_type("torch.FloatTensor")
    tmp_nc = cts_keep_filtered[cts_keep_filtered.obs.target == "NC"]

    # Filter the counts to require non-zero background counts
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
