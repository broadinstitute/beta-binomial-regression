# beta-binomial-regression

# Overview
The beta-binomial regression is a simple regression model for effect size comparisons in scRNA-seq perturb data. The method is suitable for both high and low moi perturb-seq data, including dual guide experiments. In general, the model takes in a 10x scRNA-seq output with metadata that specifies the guide call(s) for each cell, and estimates the changes in gene expression (log2FC) for each gene with respect to each guide in the experiment.

The method requires negative control guides of some form to be included in the experimental set up. We use the negative control guides to estimate the background beta distribution of each gene across the sequenced cells.

# Documentation
## The general workflow:

1. Filter your 10x scRNA-seq counts and assign guides to cells in the anndata object. Filtering steps can include mitochrondrial and ribosomal filtering, as well as minimum cell and gene filtering. In general, there should be no cells or genes with total counts = 0 in your object. Additionally, users should strive to ensure that the background cells follow this filtration rule as well (see [dual guide example](https://github.com/broadinstitute/beta-binomial-regression/blob/c59d98576523f22ddf7a5ee74eb8ce103154893e/beta-binomial-regression/dual_guide_bbr.py#L70)).
    - Key function:
        - `filter_counts()` in [count_filtering.py](https://github.com/broadinstitute/beta-binomial-regression/blob/main/beta-binomial-regression/count_filtering.py)

2. (Optional) Generate the features dataframe (and respective matrix) - a [cell x guide] matrix. This step is automatically done within the optimizer (see step 4). However, if a user would like to specify certain requirements for features (e.g. a certain features column or min number of cells required for a guide to be included), the user can generate the features matrix before running the regression.
    - Key function:
        - `generate_features_generic()` in [beta_binomial.py](https://github.com/broadinstitute/beta-binomial-regression/blob/main/beta-binomial-regression/beta_binomial.py). This function requires the current default torch tensor to be a CUDA tensor.

3. Fit the background beta distributions to the negative control counts, to get alpha and beta initial values for each gene.
    - Key function:
        - `fit_beta_binom()` in [beta_binomial.py](https://github.com/broadinstitute/beta-binomial-regression/blob/main/beta-binomial-regression/beta_binomial.py).

4. Run the regression. Because the beta-binomial regression model is independently calculated for each gene, the model can either be run on a subset of genes of interest or all genes in the study (this will take longer and use more memory). For example, if there are only 100 genes targeted in the study, but 10k genes sequenced total, the user can run the model on the 100 targeted genes alone. If the number of genes in your study exceeds 10k it would be wise to separate your data into groups of genes (could simply be 2 sets, or specific biological sets of interest) when running the model, to save memory. If the model runs out of CUDA memory, try clearing the memory and running again on a smaller set of genes.
    - Key function:
        - `sgd_optimizer()` in [beta_binomial.py](https://github.com/broadinstitute/beta-binomial-regression/blob/main/beta-binomial-regression/beta_binomial.py).

5. Get output dataframes corresponding to weights (in log2FC space) and p-values.
    - Key functions:
        - `adjust_weights_non_targeting` in [generate_bbr_outputs.py](https://github.com/broadinstitute/beta-binomial-regression/blob/main/beta-binomial-regression/generate_bbr_outputs.py)
            - Note: Run this function to adjust all targeting gene-guide weights to non-targeting weights for each gene, respectively. Non-targeting guides may have a non-zero effect on a gene, reflected in their weights estimate (if "non-targeting" is passed as a column or multiple columns in the features matrix). As the model is built on an intial estimate based on non-targeting cells, non-targeting weights act as an extra intercept that can be subtracted out for more accurate esimates of effect size and p-values, given a poor initial estimation of distributions from the non-targeting guides.
        - `make_bbr_df` in [generate_bbr_outputs.py](https://github.com/broadinstitute/beta-binomial-regression/blob/main/beta-binomial-regression/generate_bbr_outputs.py)
        - `get_pvalues_df` in [generate_bbr_outputs.py](https://github.com/broadinstitute/beta-binomial-regression/blob/main/beta-binomial-regression/generate_bbr_outputs.py)

Helpful functions that contain examples of running the model:
* `main()`: entire workflow function for dual-guide experiment in [dual_guide_bbr.py](https://github.com/broadinstitute/beta-binomial-regression/blob/main/beta-binomial-regression/dual_guide_bbr.py)
* `run_whole_bbr()`: Steps 3-5 of workflow for low-moi single-guide perturb-seq study. Contains examples of splitting/subsetting the data into smaller gene groups to save memeory. Also, contains an example of using the permutations feature. In [bbr_low_moi_example.py](https://github.com/broadinstitute/beta-binomial-regression/blob/main/beta-binomial-regression/bbr_low_moi_example.py)

## Additional Features

### Permute + Downsample
[permute_downsample.py](https://github.com/broadinstitute/beta-binomial-regression/blob/main/beta-binomial-regression/permute_downsample.py) contains the code for generating permuted and downsampled data for simulations testing. This can be used to test the method's capabilities of predicting a certain knockdown percentage in your data, by permuted guide assignments across cells and inducing a fake knockdown at a certain percentage. After generating the permuted and/or downsampled anndata object, users can run the full model on this data, by simply specifying the new column name that contains the permuted guide calls for generating the features matrix (for example, 'perm_feature_call'). `run_whole_bbr()` contains a brief example of how to specify this.

### Comparison figure
[fc_comparison_figure.py](https://github.com/broadinstitute/beta-binomial-regression/blob/main/beta-binomial-regression/fc_comparison_figure.py) contains the code for generating the simulation comparison plots. These plots aggregate the results of simulated knockdowns effect size estimates from BBR, MAST, and SCEPTRE across multiple runs. The plotting is not very scalable to other data, but nonetheless are included here for visibility.

## Notes
This method and all of its included functionality is still in its development phase. If you encounter any errors while running the model, feel free to reach out to Maddie (mamurphy@broadinstitute.org) to see if she can assist in troubleshooting as well as fix any errors in the code. This method has not yet been published.