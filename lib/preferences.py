import seaborn as sns

# ---------------------------------
# Genetic Metrics
#
# ---------------------------------


genetic_metrics = ["n_re_genomes", "n_mixed_genomes", "frac_mixed_genomes",
                   "n_re_samples", "n_mixed_samples", "frac_mixed_samples",
                   "n_fixed_ref", "n_fixed_alt",
                   "mean_k",
                   "n_barcodes", "single_barcodes", "frac_uniq_barcodes",
                   "n_variants", "n_segregating", "n_singletons",
                   "pi", "theta", "tajd"]
ibd_metrics = ["avg_frac_ibd", "avg_n_ibd", "avg_l_ibd"]
genetic_names = {"n_re_genomes": "No. Genomes",
                 "n_mixed_genomes": "No. Mixed Genomes",
                 "frac_mixed_genomes": "Frac. Mixed Genomes",
                 "n_re_samples": "No. Samples",
                 "n_mixed_samples": "No. Mixed Samples",
                 "frac_mixed_samples": "Frac. Mixed Samples",
                 "n_fixed_ref": "No. Fixed Ref. Sites",
                 "n_fixed_alt": "No. Fixed Alt. Sites",
                 "mean_k": "Complexity of Infection (Mean $k$)",
                 "n_barcodes": "No. Unique Barcodes",
                 "single_barcodes": "No. Barcodes Seen Only Once",
                 "frac_uniq_barcodes": "Frac. Barcodes Seen Only Once",
                 "n_variants": "No. Variants",
                 "n_segregating": "No. Segregating",
                 "n_singletons": "No. Singletons",
                 "pi": "Nucleotide Diversity ($ \pi $)",
                 "theta": r"Watterson's Theta ($ \theta_w $)",
                 "tajd": "Tajima's $D$",
                 "avg_frac_ibd": "Avg. Frac. IBD",
                 "avg_n_ibd": "Avg. No. IBD Tracks",
                 "avg_l_ibd": "Avg. IBD Track Length (bp)"}
genetic_alpha = 0.08
n_se = 1.96
se_alpha = 0.2

ops_metrics = ["HX", "VX", "HmX", "VmX"]
op_names = {"HX": "Host Prevalence", 
            "HmX": "Host Mixed Prevalence",
            "VX": "Vector Prevalence", 
            "VmX": "Vector Mixed Prevalence"}


def remove_metrics(rmv, metrics):
    """
    Remove `rmv` metrics from `metrics`
    """
    return [g for g in metrics if g not in rmv]


tight_metrics = ['frac_mixed_samples', 'mean_k',
                 'n_segregating', 'n_singletons',
                 'pi', 'theta', 'tajd',
                 'avg_frac_ibd', 'avg_n_ibd', 'avg_l_ibd']


# ---------------------------------
# Colours
#
# ---------------------------------


genetic_grps = {"n_re_genomes": 1,
                "n_mixed_genomes": 1,
                "frac_mixed_genomes": 1,
                "n_re_samples": 2,
                "n_mixed_samples": 2,
                "frac_mixed_samples": 2,
                "mean_k": 2,
                "n_fixed_ref": 3,
                "n_fixed_alt": 3,
                "n_barcodes": 4,
                "single_barcodes": 4,
                "frac_uniq_barcodes": 4,
                "n_variants": 5,
                "n_segregating": 5,
                "n_singletons": 5,
                "pi": 6,
                "theta": 6,
                "tajd": 6,
                "avg_frac_ibd": 7,
                "avg_n_ibd": 7,
                "avg_l_ibd": 7}

op_grps = {"HX": 8,
           "HmX": 8,
           "VX": 8,
           "VmX": 8}


def create_metric_colours(pal, genetic_metrics, genetic_grps):
    """
    Create a mapping from a set of grouped genetic metrics,
    to a colour palette
    """
    l = {}
    for metric in genetic_metrics:
        l[metric] = genetic_grps[metric]

    grps = set(l.values())
    n_grps = len(grps)

    col_pal = sns.color_palette(pal, n_grps)
    pal_grp = {g: c for g, c in zip(grps, col_pal)}

    col_dt = {}
    for metric in genetic_metrics:
        col_dt[metric] = pal_grp[l[metric]]

    return col_dt