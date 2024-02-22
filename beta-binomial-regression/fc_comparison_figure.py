import numpy as np
import pandas as pd
import seaborn as sns
import pylab
import matplotlib.pyplot as plt
import scanpy as sc

params = {'legend.fontsize': '40',
          'figure.figsize': (10, 10),
          'axes.labelsize': '40',
          'axes.titlesize': '50',
          'xtick.labelsize': '40',
          'ytick.labelsize': '40',
          'axes.linewidth': '0.5',
          'pdf.fonttype': '42',
          'font.sans-serif': 'Helvetica'}
pylab.rcParams.update(params)
plt.style.use('seaborn-white')



def make_multimethod_aggregate_plot(ax, aggregate_scores, guide, ds_counts, kd_keep, prior=None, method1=' MAST logFC', method2='',
                        method1_label='SCEPTRE logFC', method2_label='BBR Regression Weights', kd_method='Full KD', pointsize=250):
    """
    Function for plotting logFC results across multiple runs of different methods.
    This will plot the mean scores with standard deviation error bars & the target KD value in LOG2 space

    aggregate_scores: dataframe with multiindexing for guide BBR and MAST/SCEPTRE/OTHER log2FC scores.
            Each score type has mean and std (across multiple runs)

    guide: string (HNF1B, SOX2, YBX1 or HHEX e.g.) corresponding to the names of the columns

    method2 should be BBR

    """

    high_count = False
    df = aggregate_scores.unstack()

    markers, caps, bars = ax.errorbar(x=df[f'{guide}{method1}']['mean'],
                 y=df[f'{guide}{method2}']['mean'],
                 ls='none', xerr=df[f'{guide}{method1}']['std'],
                 yerr=df[f'{guide}{method2}']['std'], fmt='none')
    [bar.set_alpha(0.5) for bar in bars]

    # high_count code is a bit older, for runs where the spread of gene expression is
    # so large that color bars get messed up if plotting normally
    if (aggregate_scores['mean_TPM']['mean'] > 5000).any():
        high_count = True
        df = aggregate_scores[(aggregate_scores[f'mean_TPM']['mean'] < 5000)]
        df_2 = aggregate_scores[(aggregate_scores[f'mean_TPM']['mean'] > 5000)]
        df_2 = df_2.unstack()
        df = df.unstack()

    if high_count:
        g = sns.scatterplot(x=df[f'{guide}{method1}']['mean'],
                     y=df[f'{guide}{method2}']['mean'], hue = df['mean_TPM']['mean'], zorder=2, s=pointsize, palette="flare", ax=ax)
        b = sns.scatterplot(x=df_2[f'{guide}{method1}']['mean'],
                         y=df_2[f'{guide}{method2}']['mean'], hue = df_2['mean_TPM']['mean'], zorder=2, s=pointsize, palette='magma', ax=ax)
        handles, labels = b.get_legend_handles_labels()
        ax.legend(handles, np.hstack((labels[:-1], '5000')), fontsize=25, title_fontsize=30, title='Mean TPM')
    # Main figure plotting method:
    else:
        g = sns.scatterplot(x=df[f'{guide}{method1}']['mean'],
                     y=df[f'{guide}{method2}']['mean'], hue = np.log10(df['mean_TPM']['mean']), zorder=2, s=pointsize, palette="flare", ax=ax)

        ax.legend(fontsize=45, title_fontsize=45, title='log10(Mean TPM)')

    num_cells = ds_counts.obs.perm_working_features.value_counts()[guide]

    ax.set_ylabel(method2_label if prior is None else f'{method2_label}\n Prior = {prior}', fontsize=50)
    ax.set_xlabel(f'{method1_label}', fontsize=60)
    ax.set_title(f'{num_cells} Cells, {kd_method}', fontsize=60)

    # add a known effect target data point
    ax.axline((0, 0), slope=1, color='black')
    ax.plot(np.log2(kd_keep), np.log2(kd_keep), 'go', markersize=12)
    ax.axhline(np.log2(kd_keep), color='g', ls='--', lw=5, zorder=1)
    ax.axvline(np.log2(kd_keep), color='g', ls='--', lw=5, zorder=1)
    ax.set_aspect('equal', adjustable='datalim')

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # Add figure guide lines and shading
    x_sign = np.arange(-5, 5, 0.1)
    y_sign = np.arange(0, 5, 0.1)
    ax.fill_between(x_sign, 0, 1, alpha=0.25, facecolor='slategrey')
    ax.fill_between(y_sign, -5, 0, alpha=0.25, facecolor='slategrey')

    ax.axhline(0, color='slategrey', lw=4, zorder=1)
    ax.axvline(0, color='slategrey', lw=4, zorder=1)

    # Shade regions
    x = np.linspace(-3, 3, 100)
    y = x
    y2 = -x + 2*np.log2(kd_keep)
    ax.fill_between(x, y, y2, color='olivedrab', alpha=0.1)
    maxy1y2 = np.max([y, y2], axis=0)
    miny1y2 = np.min([y,y2], axis=0)
    ax.fill_between(y2, maxy1y2, maxy1y2.max(), color='darkmagenta', alpha=0.1)
    ax.fill_between(y2, miny1y2, miny1y2.min(), color='darkmagenta', alpha=0.1)

    return xmin, xmax, ymin, ymax


def read_bbr_sceptre_mast_results(run_num, kd_percentage, kd_type, ds_type, prior='p5'):
    """
    User would want to edit code for their specific uses. I keep simulated KD results stored by percentage, type, and analysis method.
    Then, read in all the results.
    e.g.
    "D0_perm_ds_50/full_kd/ds_100_uniform_sampling/sceptre_results/sceptre_low_moi_50_kd_1_discovery.csv"

    """
    sceptre_results_table = pd.read_csv(f'D0_perm_ds_{kd_percentage}/{kd_type}/{ds_type}/sceptre_results/sceptre_lowmoi_{kd_percentage}_kd_{run_num}_discovery.csv')
        # Prior 0.5 data
    file_lab = ds_type.split('_')[1] + '_genes'
    #print(f'D0_perm_ds_{kd_percentage}/{kd_type}/{ds_type}/bbr_results/ds_{file_lab}_results_{prior}_{run_num}.csv')
    bbr_results = pd.read_csv(f'D0_perm_ds_{kd_percentage}/{kd_type}/{ds_type}/bbr_results/ds_{file_lab}_results_{prior}_{run_num}.csv', index_col=0).set_index('gene')
    sceptre_logfc = sceptre_results_table.set_index('response_id')[['grna_group', 'log_2_fold_change']].pivot(columns='grna_group')['log_2_fold_change']
    sceptre_pvalue = sceptre_results_table.set_index('response_id')[['grna_group', 'p_value']].pivot(columns='grna_group')['p_value']
    sceptre_logfc = bbr_results[['mean_TPM']].merge(sceptre_logfc, left_index=True, right_index=True)

    mast_results = pd.read_csv(f'D0_perm_ds_{kd_percentage}/{kd_type}/{ds_type}/MAST_results/ds_100_mast_result_{run_num}.csv')
    mast_logfc = mast_results[['gene', 'logFC', 'Guide']].pivot(index='gene', columns='Guide')['logFC'].add_suffix('_MAST')

    return sceptre_logfc, sceptre_pvalue, bbr_results, mast_logfc


def make_aggregate_df(KD_TYPE, percent, ds_type, threshold_3=False, pval_filter=False):
    """
    Read in all simulated KD results from each method across 10 runs, and make an aggregate df.

    Again, generic user can't use this code as it's specific to our data.
    But, the code for making the aggregate df can be used.

    """
    file_lab = ds_type.split('_')[1] + '_genes'

    sceptre_logfcs = {}
    sceptre_pvalues = {}
    bbr_results_tables = {}
    merged_dfs = {}
    mast_logfc = {}
    compare_cols = 'YBX1,SOX2,GATA3,HHEX,ID1,HAND1,TBXT,HNF1B'.split(',')

    for num in np.arange(1,11):
        sceptre_logfcs[num], sceptre_pvalues[num], bbr_results_tables[num], mast_logfc[num] = read_bbr_sceptre_mast_results(num, percent, KD_TYPE, ds_type, prior='p1.0')

    # generate a pvalue mask for sceptre, requiring at least 5 significant pvalues
    sceptre_gene_mask_5_sig = (pd.concat(sceptre_pvalues) < 0.05).reset_index(1).groupby('response_id').sum() >=5
    # generate value mask for sceptre, which requires just at least 3 non-nan returns for a gene
    sceptre_gene_mask_3_nonnan = (pd.concat(sceptre_logfcs).reset_index().groupby('level_1').count() >= 3).drop(columns=['level_0', 'mean_TPM'])

    for num in np.arange(1,11):
        assert sceptre_logfcs[num].columns[0] == 'mean_TPM'
        if pval_filter:
            # filter sceptre information based on pvalues
            sceptre_logfcs[num].iloc[:, 1:] = sceptre_logfcs[num].iloc[:, 1:].where(sceptre_gene_mask_5_sig, np.nan)
        # TODO: decide if this is elif or if--do we want these to be separate options or can a user possibly do both?
        elif threshold_3:
            sceptre_logfcs[num].iloc[:, 1:] = sceptre_logfcs[num].iloc[:, 1:].where(sceptre_gene_mask_3_nonnan, np.nan)
        sceptre_bbr_merge = sceptre_logfcs[num].merge(bbr_results_tables[num][compare_cols], left_index=True, right_index=True, suffixes=('_sceptre', '_bbr'))
        merged_dfs[num] = sceptre_bbr_merge.merge(mast_logfc[num], left_index=True, right_index=True)

    stacked_df = pd.concat(merged_dfs, ignore_index=False).reset_index(1).rename(columns={'level_1':'gene'})
    aggregate_bbr_sceptre = stacked_df.groupby('gene').agg([np.mean, np.std])

    # Get one of the actual knocked down counts objects for TPM calcs.
    # TODO: Would be more efficient to just get mean TPM earlier in pipeline, so users don't need to read in h5ad.
    one_kd_counts = sc.read_h5ad(f'D0_perm_ds_{percent}/{KD_TYPE}/{ds_type}/ds_{file_lab}_1.h5ad')


    return aggregate_bbr_sceptre, one_kd_counts


def generate_figure():
    """
    Because each plot can differ greatly in number of subplots, types of comparisons (num cells, kd percentage, kd type),
        users should just edit this code based on their specific needs/wants out of a plot.
        As of now, there isn't a super easy way to scale this up, but users can see how to call "make_multimethod_aggregate_plot" to get desired subplots.
        And use this in any combination they desire.
    """

    # plotting example with sceptre pvalue filter on
    aggregate_bbr_sceptre_50, one_kd_counts_50 = make_aggregate_df(KD_TYPE='full_kd', percent='50', ds_type='ds_100_uniform_sampling', pval_filter=True)
    aggregate_bbr_sceptre_25, one_kd_counts_25 = make_aggregate_df(KD_TYPE='full_kd', percent='25', ds_type='ds_100_uniform_sampling', pval_filter=True)
    aggregate_bbr_sceptre_15, one_kd_counts_15 = make_aggregate_df(KD_TYPE='full_kd', percent='15', ds_type='ds_100_uniform_sampling', pval_filter=True)

    fig, axs = plt.subplots(1, 4, figsize=(60, 15), constrained_layout=True)
    # this is the matplotlib 3.5 version for setting pads
    fig.set_constrained_layout_pads(hspace=0.1)

    # plot 1
    # need to reset x and ylim each time because it's really sensitive
    # pass kd_keep for the KD percentage you're plotting
    xmin, xmax, ymin, ymax = make_multimethod_aggregate_plot(axs[0], aggregate_bbr_sceptre_50, 'SOX2', one_kd_counts_50, kd_keep=.5,
                        prior=1, method1='_sceptre', method2='_bbr', method1_label='SCEPTRE logFC', kd_method='50% All Cell KD')
    axs[0].set_aspect('auto')
    axs[0].set_xlim(xmin,xmax)
    axs[0].set_ylim(ymin,ymax)

    # plot 2
    xmin, xmax, ymin, ymax = make_multimethod_aggregate_plot(axs[1], aggregate_bbr_sceptre_25, 'SOX2', one_kd_counts_25, kd_keep=.75,
                        prior=1, method1='_sceptre', method2='_bbr', method1_label='SCEPTRE logFC', kd_method='25% All Cell KD')
    axs[1].set_aspect('auto')
    axs[1].set_xlim(xmin,xmax)
    axs[1].set_ylim(ymin,ymax)

    # plot 3
    xmin, xmax, ymin, ymax = make_multimethod_aggregate_plot(axs[2], aggregate_bbr_sceptre_15, 'SOX2', one_kd_counts_15, kd_keep=.85,
                        prior=1, method1='_sceptre', method2='_bbr', method1_label='SCEPTRE logFC', kd_method='15% All Cell KD')
    axs[2].set_aspect('auto')
    axs[2].set_xlim(xmin,xmax)
    axs[2].set_ylim(ymin,ymax)

    # plot 4
    xmin, xmax, ymin, ymax = make_multimethod_aggregate_plot(axs[3], aggregate_bbr_sceptre_15, 'HNF1B', one_kd_counts_15, kd_keep=.85,
                        prior=1, method1='_sceptre', method2='_bbr', method1_label='SCEPTRE logFC', kd_method='15% All Cell KD')
    axs[3].set_aspect('auto')
    axs[3].set_xlim(xmin,xmax)
    axs[3].set_ylim(ymin,ymax)

    # labeling and title
    handles, labels = axs[2].get_legend_handles_labels()
    for ax in axs.ravel():
        ax.set_ylabel('')
        ax.get_legend().remove()

    fig.supylabel('BBR Regression Weights\n Prior=1', fontsize=65, ha='center')
    fig.suptitle('Simulated Knockdown Method Comparison, \nwith 5 significant sceptre pvalues thresholding', fontsize=70, y=1.15)
    fig.legend(handles, labels, bbox_to_anchor=(1.1,.6), title='log10(Mean TPM)', fontsize=40, title_fontsize=45, markerscale=2)

    return fig

def main():
    return generate_figure()

if __name__ == '__main__':
    main()