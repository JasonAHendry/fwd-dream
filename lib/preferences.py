import seaborn as sns
import numpy as np
import matplotlib as mpl


# ================================================================= #
# Metric Names
#
#
# ================================================================= #


# Names for prevalence statistics
op_names = {"H1": "No. Infected Hosts, Total",
            "Hm": "No. Infected Hosts, Mixed",
            "HX": "Host Prevalence",
            "HmX": "Host Mixed Prevalence",
            "V1": "No. Infected Vectors, Total",
            "Vm": "No. Infected Vectors, Mixed",
            "VX": "Vector Prevalence", 
            "VmX": "Vector Mixed Prevalence"}


# Names for genetic diversity metrics
genetic_names = {
    "n_samples": "No. Samples",
    "n_genomes": "No. Genomes",
    "n_mixed_samples": "No. Mixed Samples",
    "frac_mixed_samples": "Frac. Mixed Samples",
    "mean_k": "Mean COI",
    "max_k": "Max. COI",
    "n_variants": "No. Variants",
    "n_segregating": "No. Segregating",
    "n_singletons": "No. Singletons",
    "pi": "Nucleotide Diversity ($ \pi $)",
    "theta": r"Watterson's Theta ($ \theta_w $)",
    "tajd": "Tajima's $D$"
}

# IBD related
ibd_metrics = [prefix + suffix 
               for prefix in ["f_ibd", "l_ibd", "f_ibs", "l_ibs"] 
               for suffix in ["", "_bw", "_wn"]]
ibd_names = {
    "f_ibd": "Fraction IBD",
    "f_ibs": "Fraction IBS",
    "l_ibd": "Mean IBD Segment Length",
    "l_ibs": "Mean IBS Segment Length",
}
ibd_within = {k + "_wn" : v + " Within Host" for k, v in ibd_names.items()}
ibd_between = {k + "_bw" : v + " Between Hosts" for k, v in ibd_names.items()}
ibd_names.update(ibd_within)
ibd_names.update(ibd_between)

genetic_names.update(ibd_names)


      
# ================================================================= #
# Selecting Metrics
#
#
# ================================================================= #


tight_metrics = ['frac_mixed_samples', 'mean_k',
                 'n_segregating', 'n_singletons',
                 'pi', 'theta', 'tajd',
                 'f_ibd', 'l_ibd', 'f_ibs', 'l_ibs']
    
    
def remove_metrics(rmv, metrics):
    """
    Remove `rmv` metrics from `metrics`
    """
    return [g for g in metrics if g not in rmv]


# ================================================================= #
# Colours and groupings for metrics
#
#
# ================================================================= #


prevalence_cols = ['#d73027','#fc8d59','#2166ac', '#67a9cf']
prevalence_col_dt = dict(zip(["HX", "HmX", "VX", "VmX"], prevalence_cols))

def create_shades(color, n, base="white"):
    """
    Create shades of a color
    """
    
    color = np.array(mpl.colors.to_rgb(color))
    base = np.array(mpl.colors.to_rgb(base))
    fs = np.arange(n) / n
    shades = [(1 - f) * color + f * base for f in fs]
    
    return [mpl.colors.to_hex(s) for s in shades]

tight_metric_cols = sns.color_palette("viridis", len(tight_metrics))
tight_metric_col_dt = dict(zip(tight_metrics, 
                               tight_metric_cols))


# GROUPS (for colouring by group)
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

