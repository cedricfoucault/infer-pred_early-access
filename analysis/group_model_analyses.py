"""
Perform analyses comparing participant and model behavior.
"""

from analyses import save_analysis_stats
import argparse
import infer_pred_model
import numpy as np
import os.path as op
import pandas as pd
import scipy.stats
import sub_data
import utils

def main(args):
    np.random.seed(0)
    dataset = args.dataset
    sub_list_path, task_config_path = utils.datasets.get_dataset_paths(dataset)
    data_files = sub_data.get_subject_files(sub_list_path)
    task_config = utils.config.load_config_at_path_with_default_task_config(task_config_path)
    models = []
    configs = []
    model_ids = []
    for mcp in args.model_config_paths:
        model_config = utils.config.load_config_at_path_with_default_model_config(mcp)
        config = utils.config.create_joint_config(task_config, model_config)
        model = infer_pred_model.get_model(config)
        models += [model]
        configs += [config]
        model_ids += [model.config["model"]["id"]]

    df, suffixes = build_joined_subject_model_df(data_files, configs, models)
    sub_ids = sorted(df["subjectId"].unique())

    # Trial-wise learning rate correlations
    out_params = ["models"], ["-".join(model_ids)]
    outdirname = dataset
    analysis_name = "learning_rate/human_model_comparison"
    corrs = compute_trial_lr_corrs(df, suffixes, sub_ids)
    stats_data = compute_stats(corrs, model_ids=model_ids)
    save_analysis_stats(stats_data, out_params, analysis_name, outdirname=outdirname,
                        prefix="human_model_correlation_comparison")
    # Also save the per-subject correlation matrix for inspection
    if args.do_save_correlation_values:
        corr_df = pd.DataFrame(corrs, index=sub_ids, columns=model_ids)
        save_analysis_stats(corr_df, out_params, analysis_name, outdirname=outdirname,
            prefix="human_model_correlation_values")

def build_joined_subject_model_df(data_files, configs, models):
    """
    Build a single dataframe per subject that joins the subject trial data
    with each model's simulated trial data, then concatenate across subjects.
    Returns:
      df: concatenated dataframe across subjects
      suffixes: list like ["", "_model_1", "_model_2", ...] used in column names
    Notes:
      - The subject dataframe uses configs[0] (task config is shared across models).
    """
    sub_dfs = []
    model_suffixes = [f"_model_{i+1}" for i in range(len(models))]
    suffixes = [""] + model_suffixes
    join_cols = ['subjectId', 'blockIdx', 'trialIdx']
    for data_file in data_files:
        data = sub_data.load_subject_data(data_file)
        # Subject dataframe (no model) — uses first config for task-dependent fields
        df = sub_data.make_dataframe_for_subject_data_and_model(data, configs[0], model=None)
        # Merge each model's dataframe onto the subject dataframe
        for i_model, model in enumerate(models):
            df_model = sub_data.make_dataframe_for_subject_data_and_model(
                data, configs[i_model], model=model
            )
            # all_cols = ['blockIdx', 'trialIdx', 'hmean', 'hsd', 'obs', 'betloc',
            #     'betwid', 'score', 'subjectId', 'ape', 'updt_loc', 'lr', 'p_updt_loc',
            #     'lr_exclude-no-update', 'plrgwid', 'nextplrgwid', 'updt_wid', 'pincwid',
            #     'pdecwid', 'hmeanchg', 'hsdchg', 'prevhsd', 'did_catch', 'mae',
            #     'reward']
            df = pd.merge(
                df,
                df_model,
                on=join_cols,
                suffixes=(suffixes[0], suffixes[i_model + 1]),
            )
        sub_dfs.append(df)
    out_df = pd.concat(sub_dfs, ignore_index=True)
    return out_df, suffixes

def compute_trial_lr_corrs(df, suffixes, sub_ids):
    """
    Returns array of correlation values between each subject and each model's
    trial-wise learning rates.
    Shape: (n_subjects, n_models)
    """
    corrs = np.empty((len(sub_ids), len(suffixes) - 1))
    for i in range(1, len(suffixes)):
        suffix_pair = (suffixes[0], suffixes[i])
        corr = df.groupby("subjectId")[[f"lr{suffix_pair[0]}", f"lr{suffix_pair[1]}"]].apply(
            lambda x: x[f"lr{suffix_pair[0]}"].corr(x[f"lr{suffix_pair[1]}"])
        )
        corrs[:, i - 1] = corr.loc[sub_ids].values
    return corrs


def compute_lr_ape_bin_corrs(df, suffixes, sub_ids, ape_bins):
    """
    Returns array of correlation values between subject and each model's learning rate curve
    as a function of absolute prediction error bins, for each subject.
    Shape: (n_subjects, n_models)
    """
    n_subs = len(sub_ids)
    n_models = len(suffixes) - 1
    avg_lrs_per_sub_bin = np.empty((n_subs, len(ape_bins)-1, len(suffixes)))
    for i, suff in enumerate(suffixes):
        bin_key = f"ape_bin{suff}"
        ape_key = f"ape{suff}"
        lr_key = f"lr{suff}"
        df[bin_key], bin_edges = pd.cut(df[ape_key], bins=ape_bins,
                retbins=True, labels=False)
        for i_sub, sub in enumerate(sub_ids):
            avg_lrs_per_bin = (
                df[df["subjectId"] == sub]
                .groupby(bin_key)[lr_key]
                .mean(numeric_only=True)
                .values
            )
            avg_lrs_per_sub_bin[i_sub, :, i] = avg_lrs_per_bin

    corrs = np.empty((n_subs, n_models))
    for i_model in range(n_models):
        for i_sub in range(n_subs):
            sub_lrs = avg_lrs_per_sub_bin[i_sub, :, 0]
            model_lrs = avg_lrs_per_sub_bin[i_sub, :, i_model+1]
            if np.any(np.isnan(sub_lrs)) or np.any(np.isnan(model_lrs)):
                print(f"nan for subject {i_sub}, model {i_model+1}")
                print("subject lrs: ", sub_lrs)
                print("model lrs: ", model_lrs)
            # else:
            #     corrs[i_sub, i_model] = scipy.stats.pearsonr(sub_lrs, model_lrs)[0]
            mask = ~np.isnan(sub_lrs) & ~np.isnan(model_lrs)
            corrs[i_sub, i_model] = scipy.stats.pearsonr(sub_lrs[mask], model_lrs[mask])[0]
            # corrs[i_sub, i_model] = np.sqrt(((sub_lrs - model_lrs) ** 2)[mask].mean())
    return corrs

def compute_stats(corrs, model_ids=None, do_descriptive=False):
    """
    Do statistical comparison (paired t-tests) between adjacent models' subject-wise correlation coefficients.
    corrs: np.array of shape (n_subs, n_models)
    """
    if do_descriptive:
        # Compute the average and s.e.m of the correlation coefficients
        corr_mean = np.nanmean(corrs, axis=0)
        corr_sem = scipy.stats.sem(corrs, axis=0, nan_policy='omit')
        print("Average correlation with each model:")
        print(corr_mean)
        print("S.E.M of correlations with each model:")
        print(corr_sem)
        for i_model in range(corrs.shape[1]):
            print(f"Correlation with model {i_model + 1}:")
            print("Average:", corr_mean[i_model])
            print("S.E.M:", corr_sem[i_model])
    # Perform a paired samples t-test of the correlation coefficients between two models
    cols = []
    t_vals = []
    p_vals = []
    dof_vals = []
    mean_diffs = []
    median_diffs = []
    d_vals = []
    binom_p_vals = []
    binom_ci_lower_bounds = []
    binom_ci_upper_bounds = []
    props_better = []
    for i_model in range(corrs.shape[1] - 1):
        x = corrs[:, i_model]
        y = corrs[:, i_model + 1]
        # Drop subjects with nan in either column for a fair paired comparison
        mask = np.isfinite(x) & np.isfinite(y)
        ttest = scipy.stats.ttest_rel(x[mask], y[mask], nan_policy="omit")
        t_vals += [ttest.statistic]
        p_vals += [ttest.pvalue]
        dof_vals += [ttest.df]
        diff = (x[mask] - y[mask])
        mean_diff = diff.mean()
        d = mean_diff / diff.std(ddof=1) # Cohen's d
        mean_diffs += [mean_diff]
        median_diffs += [np.median(diff)]
        d_vals += [d]
        x_better = (x[mask] > y[mask])
        props_better += [x_better.sum() / x_better.size]
        binom = scipy.stats.binomtest(x_better.sum(), x_better.size)
        binom_p_vals += [binom.pvalue]
        binom_prop_ci = binom.proportion_ci()
        binom_ci_lower_bounds += [binom_prop_ci[0]]
        binom_ci_upper_bounds += [binom_prop_ci[1]]
        cols += [(f"Model {i_model + 1} vs model {i_model + 2}" if model_ids is None
                    else f"{model_ids[i_model]} vs {model_ids[i_model+1]}")]
    return pd.DataFrame.from_dict(
        {
            "t statistic": t_vals,
            "p value": p_vals,
            "degrees of freedom": dof_vals,
            "mean difference in correlation": mean_diffs,
            "median difference in correlation": median_diffs,
            "Cohen's d": d_vals,
            "proportion of participants better matched by first model": props_better,
            "CI lower bound for the proportion (binomial test)": binom_ci_lower_bounds,
            "CI upper bound for the proportion (binomial test)": binom_ci_upper_bounds,
            "binomial test p value": binom_p_vals,
        },
        orient="index",
        columns=cols,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["change-point-env", "random-walk-env"],
        help="Which experimental dataset / environment to analyze.")
    parser.add_argument("-mcps", "--model_config_paths", nargs='+', required=True)
    parser.add_argument("--do_save_correlation_values", action="store_true")
    args = parser.parse_args()
    main(args)
