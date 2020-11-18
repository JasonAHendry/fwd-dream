import random
import numpy as np
import itertools
import allel
import scipy.stats
import configparser


# ================================================================= #
# Sequencing
#
# ================================================================= #


def detect_mixed(oo, detection_threshold=None):
    """
    Detect if a given infection is mixed

    Parameters
        oo: ndarray, shape (nph, nsnps)
            2D Array containing binary genomes for given
            organism (host or vector). Rows are genomes,
            columns are SNPs.
        detection_threshold: float
            From [0, 1], specifies the fraction of SNPs
            that must be heterozygous for an infection
            to be detected as mixed.
    Return
        ans: bool
            Answer, is the infection mixed or not?
    """
    n_het_sites = (oo.min(0) != oo.max(0)).sum()
    if detection_threshold is None:
        ans = n_het_sites > 0
    else:
        n_snps = oo.shape[1]
        ans = n_het_sites / float(n_snps) >= detection_threshold
    return ans


def sequence_dna(hh, detection_threshold=None):
    """
    Sequence parasite genomes within a single host
    reporting `k`, the number of strains present
    and `seqs`, the sequences themselves

    Note that identical repeated parasite genomes are
    reported only once by `sequence_dna()`

    Parameters
        hh: ndarray, shape (nph, nsnps)
            Array containing parasite genomes for single host.
        detection_threshold: None or float
            A haplotype is only sequenced if > `detection_threshold`
            of its sites are heterozygous, when compared to *all*
            other collected genomes.

    Returns
        k: int
            Number of unique parasite genomes within the
            sequenced host.
        seqs: ndarray, shape(k, nsnps)
            Sequenced parasite genomes.
    """
    nsnps = hh.shape[1]
    seqs = np.vstack(list({tuple(row) for row in hh}))
    k = seqs.shape[0]
    if k > 1 and detection_threshold is not None:  # now we check for enough differences
        keep_indxs = [0]
        for j in np.arange(1, k):
            ndiffs = (seqs[keep_indxs, :] != seqs[j]).sum(1)
            if (ndiffs / nsnps >= detection_threshold).all():
                keep_indxs.append(j)
        seqs = seqs[keep_indxs]
        k = seqs.shape[0]
    return k, seqs


def collect_genomes(h_dt, max_samples=None, detection_threshold=None, verbose=False):
    """
    Collect parasite genomes from a population of infected
    human beings

    Parameters
        h: ndarray, shape (nh)
            Infection status (0/1) of each host in the population
            of size `nh`.
        h_a: ndarray, shape (nh, nph, nsnps)
            3D array holding parasite genomes for all hosts.
        max_samples: None or int
            Desired maximum number of samples to collect.
        detection_threshold: None or float
            A haplotype is only collected if > `detection_threshold`
            of its sites are heterozygous, when compared to *all*
            other collected genomes.
        verbose: bool
    Returns
        ks: ndarray, shape (samples)
            Number of unique parasite genomes for each of the samples
            that were sequenced.
        genomes: ndarray, shape (nsnps, samples)
            All parasite genomes collected.
    """
    
    # Determine how many samples to collect
    n_infected = len(h_dt.keys())
    if max_samples is None or max_samples >= n_infected:
        n_collect = n_infected
    else:
        n_collect = max_samples
    
    # Collect and sequence DNA of samples
    ixs = []
    ks = []
    gs = []
    for idc in random.sample(h_dt.keys(), k=n_collect):
        k, g = sequence_dna(hh=h_dt[idc], detection_threshold=detection_threshold)
        ks.append(k)
        gs.append(g)
        ixs.extend([idc]*k)  # sample indices for each genome collected

    # Convert to numpy arrays
    genomes = np.vstack(gs).T  # transpose so rows=snps
    ks = np.array(ks)
    ixs = np.array(ixs)
    
    return ks, genomes


def get_allele_counts(genomes):
    """
    Generate an allele count array
    for a collection of genomes
    
    Parameters
        genomes: ndarray, shape (nsnps, n_genomes)
            Array encoding a set of sequenced parasite
            genomes.
    
    Returns:
        ac: AlleleCountArray, shape (nsnps, n_alleles)
            Allele counts for every loci in `genomes`.
            
    """
    nsnps, ngenomes = genomes.shape
    ac = np.zeros((nsnps, ngenomes), 'int16')  # the maximum possible size
    for i in np.arange(nsnps):
        counts = np.unique(genomes[i], return_counts=True)[1]
        n = len(counts)
        ac[i, :n] = counts
    ac = ac[:, ac.sum(0) > 0]  # remove columns with no alleles
    return allel.AlleleCountsArray(ac)


def gen_allel_datastructs(genomes):
    """
    Generate scikit-allel data structures
    (genomes, AlleleCountArray, and a vector
    of SNP positions) from sequenced parasite genomes

    Parameters
        genomes: ndarray, shape (nsnps, n_genomes)
            Array encoding a set of sequenced parasite
            genomes.

    Returns
        genomes: ndarray, shape (nsnps, n_genomes)
            Array encoding a set of sequenced parasite
            genomes. for each parasite genome,
            where ref=0 and alt=1.
        ac: AlleleCountArray, shape (nsnps, 2)
            Allele counts computed from `hap`.
        pos: ndarray, shape (nsnps)
            The position, as an integer, of each SNP
            in the parasite genome.
    """
    # hap = allel.HaplotypeArray(genomes), no longer possible
    ac = get_allele_counts(genomes)
    pos = np.arange(1, genomes.shape[0] + 1)
    return genomes, pos, ac


def calc_k_stats(ks, verbose=False):
    """
    Calculate a suite of statistics on the distribution
    of the number of unique genomes per host

    Parameters
        ks: ndarray, shape(n_hosts)
            The number of unique parasite genomes
            discovered in each sequenced host.
        verbose: bool
    Returns
        k_stats: dict
            Dictionary containing statistics.
    """
    n_re_genomes = ks.sum()
    n_mixed_genomes = ks[ks > 1].sum()
    n_re_samples = len(ks)
    n_mixed_samples = len(ks[ks > 1])
    frac_mixed_genomes = float(n_mixed_genomes) / n_re_genomes
    frac_mixed_samples = float(n_mixed_samples) / n_re_samples
    mean_k = ks.mean()
    if verbose:
        print("Genomes")
        print("  No.:", n_re_genomes)
        print("  Mixed:", n_mixed_genomes)
        print("  %:", frac_mixed_genomes * 100)
        print("Samples")
        print("  No.:", n_re_samples)
        print("  Mixed.:", n_mixed_samples)
        print("  %:", frac_mixed_samples * 100)

    k_stats = {"ks": ks.tolist(),
               "n_re_genomes": n_re_genomes,
               "n_mixed_genomes": n_mixed_genomes,
               "n_re_samples": n_re_samples,
               "n_mixed_samples": n_mixed_samples,
               "frac_mixed_genomes": frac_mixed_genomes,
               "frac_mixed_samples": frac_mixed_samples,
               "mean_k": mean_k}
    return k_stats


def get_barcodes(genomes, verbose=False):
    """
    Get `barcodes` (=unique genomes/haplotypes)
    from the parasite haplotype array

    `barcode` terminology is in reference to Daniels et al.
    PNAS, 2015, in which they calculate similar statistics
    on SNP barcodes to track malaria prevalence changes.

    Parameters
        genomes: ndarray, shape (nsnps, n_genomes)
            Array encoding a set of sequenced parasite
            genomes.
    Returns
        barcode_counts: ndarray, shape (n_unique_genomes)
            Returns the counts for each unique genome in the
            population.
    """
    barcode_counts = np.unique(genomes, axis=1, return_counts=True)[1]
    n_barcodes = len(barcode_counts)
    if verbose:
        print("Senegal Statistics")
        print("  No. Unique Barcodes:", n_barcodes)
        print("  No. Barcodes Observed Only Once:", (barcode_counts == 1).sum())
        print("  ... No. Times a Given Barcode is Observed")
        print("  Min.:", barcode_counts.min())
        print("  Mean:", barcode_counts.mean())
        print("  Median:", np.median(barcode_counts))
        print("  Max.:", barcode_counts.max())
    return barcode_counts


def calc_diversity_stats(pos, ac, verbose=False):
    """
    Calculate a suite of standard genetic diversity
    statistics given SNP positions and allele counts,
    for a set of parasite genomes

    Parameters
        ac: AlleleCountArray, shape (nsnps, 2)
            Allele counts computed from `hap`.
        pos: ndarray, shape (nsnps)
            The position, as an integer, of each SNP
            in the parasite genome.
    Returns
        div_stats: dict
            Dictionary of genetic diversity statistics.
    """
    _, n_allele_types = ac.shape
    if n_allele_types == 1:  # population clonal, f(x)'s would fail
        n_segregating = 0
        n_singletons = 0
        statement = "  Sample is clonal."
    else:
        n_segregating = ac.count_segregating()
        n_singletons = (ac == 1).any(1).sum()
        statement = "  Sample is not clonal."

    pi = allel.stats.diversity.sequence_diversity(pos, ac)
    theta = allel.stats.diversity.watterson_theta(pos, ac)
    tajd = allel.stats.diversity.tajima_d(ac)  # <4 samples, throws warning (NOT exception)

    if verbose:
        print("Diversity Metrics")
        print(statement)
        print("  No. Segregating Sites:", n_segregating)
        print("  No. Singletons:", n_singletons)
        print("  Nucleotide Diversity (pi):", pi)
        print("  Watterson's Theta:", theta)
        print("  Tajima's D:", tajd)

    div_stats = {"n_segregating": n_segregating,
                 "n_singletons": n_singletons,
                 "pi": pi,
                 "theta": theta,
                 "tajd": tajd}

    return div_stats


def get_ibd_segments(ibd):
    """
    Given a boolean vector specifying whether
    each SNP position carries the same allele,
    return an array of identity track lengths
    
    Parameters
        ibd: ndarray, dtype bool, shape (nsnps)
            For each position in the genome, are
            the alleles identical?
    Returns
        segs: ndarray, dtype int, shape(n_segs)
            Return the length of IBD segments.
    """
    ll = []
    l = 0
    for state in ibd:
        if state:
            l += 1
        else:
            ll.append(l)
            l = 0
    ll.append(l)  # append last segment
    segs = np.array(ll)  # always positive, won't be >65K SNPs

    return segs[segs > 0]  # only return if positive length


def calc_ibd_statistics(genomes):
    """
    For every pair of genomes collected, return
    three IBD summary statistics: 
     - IBD fraction
     - Mean IBD track length
     - Number of IBD segments

    Here, identity-by-descent (IBD) is approximated 
    with identity-by-state (IBS). However, in an 
    infinite-alleles model, IBS only occurs through
    IBD as the same allele is never generated twice.
    Thus if two samples carry the same allele,
    it is through common inheritance.
    
    
    Parameters
        genomes: ndarray, shape (nsnps, ngenomes)
            Array encoding a set of sequenced parasite
            genomes.
    
    Returns
        f_ibd: ndarray, float, shape (n_pairs)
            Fraction IBD between every possible pair
            of genomes. Length is the number of
            pairs, i.e. n choose 2.
        n_ibd: ndarray, int, shape (n_pairs)
            Number of IBD segments between every
            possible pair of genomes.
        l_ibd: ndarray, float, shape (n_pairs)
            Mean length of IBD segments between
            every possible pair of genomes.
    """
    nsnps, ngenomes = genomes.shape
    
    # Prepare storage
    n_pairs = int(ngenomes * (ngenomes - 1) / 2)
    f_ibd = np.zeros(n_pairs)
    l_ibd = np.zeros(n_pairs)
    n_ibd = np.zeros(n_pairs)
    
    # Loop over all pairs of genomes
    pairs = itertools.combinations(range(ngenomes), 2)
    for ix, (i, j) in enumerate(pairs):
        
        # Compute IBD state and segments
        ibd_state = genomes[:, i] == genomes[:, j]
        ibd_segments = get_ibd_segments(ibd_state)
        
        # Compute IBD summary statistics
        f_ibd[ix] = ibd_state.sum()
        n_ibd[ix] = ibd_segments.shape[0]
        l_ibd[ix] = ibd_segments.mean()
        
    f_ibd /= nsnps
    
    return f_ibd, n_ibd, l_ibd


def calc_unfolded_sfs(ac, n, verbose=False):
    """
    Compute the *full* unfolded site-frequency spectrum
    given an allele count array and the number of genomes

    In some instances, the simulation parasite population
    is clonal, in which case the SFS does not exist and
    `-1` is returned for all derived allele counts.

    Parameters
        ac: AlleleCountArray, shape (nsnps, 2)
            Allele counts computed from `hap`.
        n: int
            Number of genomes collected
        verbose: bool
    Returns
        unfolded_sfs: ndarray, shape (n)
            The unfolded SFS; the count of the number of
            SNPs with each possible number of derived
            alleles.
    """
    if verbose:
        print("Computing unfolded SFS.")

    _, n_allele_types = ac.shape
    full_unfolded_size = n + 1
    no_fixed_size = full_unfolded_size - 2
    if n_allele_types == 1:  # clonal, no SFS
        if verbose:
            print("Population is clonal. No SFS exists.")
        sfs_unfolded = np.empty(no_fixed_size)
        sfs_unfolded.fill(-1)  # signifies NaN, but preserves dtype
    else:
        sfs_unfolded = allel.sfs(ac[:, 1])  # will include fixed if complete
        if len(sfs_unfolded) < full_unfolded_size:
            if verbose:
                print("  Unfolded SFS was truncated to %d sites." % len(sfs_unfolded))
                print("  Extending by %d sites..." % (full_unfolded_size - len(sfs_unfolded)))
            sfs_unfolded = np.concatenate((sfs_unfolded, np.zeros(full_unfolded_size - len(sfs_unfolded))))
        sfs_unfolded = sfs_unfolded[1:-1]  # remove fixed ref and fixed alt
    assert len(sfs_unfolded) == no_fixed_size
    if verbose:
        print("  SFS:", sfs_unfolded[:np.min((4, n))], "...")

    return sfs_unfolded


def calc_r(x, y):
    """
    Calculate the correlation co-efficient between
    two 1D numpy arrays
    """
    x_ = x - x.mean()
    y_ = y - y.mean()
    num = np.dot(x_, y_)
    sd_x = np.sqrt(np.dot(x_, x_))
    sd_y = np.sqrt(np.dot(y_, y_))
    return num/(sd_x*sd_y)


def calc_r2_decay(hap, pos, ac, lowest_freq=0.15, verbose=False):
    """
    Compute pairwise linkage disequilibrium between all
    SNPs with a MAF >= `lowest_freq`
    
    Parameters
        hap: HaplotypeArray, shape (nsnps, n_genomes)
            Array of allele indicies for each parasite genome,
            where ref=0 and alt=1.
        ac: AlleleCountArray, shape (nsnps, 2)
            Allele counts computed from `hap`.
        pos: ndarray, shape (nsnps)
            The position, as an integer, of each SNP
            in the parasite genome.
        lowest_freq: float
            The lowest permissible MAF for SNPs included.
        verbose: bool
    Returns
        r2: ndarray, shape (n_pairs)
            Array of squared correlation coefficients
            between each pair of SNPs.
        d: ndarray, shape (n_pairs)
            Array of distances between each pair of
            SNPs.
    """

    n_genomes = ac[0, :].sum()
    maf = ac.min(1) / float(n_genomes)
    pos = pos[maf >= lowest_freq]
    hap = hap[maf >= lowest_freq]
    n_sites = len(pos)

    if verbose:
        print("Pairwise LD decay (r2)")
        print("  No. Genomes:", n_genomes)
        print("  Lowest MAF:", lowest_freq)
        print("  No. Sites > MAF:", len(pos))

    if n_sites >= 2:
        n_pairs = int(n_sites * (n_sites - 1) / 2)
        r2 = np.zeros(n_pairs)
        d = np.zeros(n_pairs, dtype='uint16')

        ix = 0
        for i in np.arange(n_sites):
            for j in np.arange(i + 1, n_sites):
                r2[ix] = calc_r(hap[i], hap[j])   # rows are mutant sites
                d[ix] = np.abs(pos[j] - pos[i])
                ix += 1

        r2 **= 2

        if verbose:
            print("  No. pairs:", len(r2))
            print("  Min. LD:", r2.min())
            print("  Mean LD:", r2.mean())
            print("  Median LD:", np.median(r2))
            print("  Max. LD:", r2.max())
            print("  Min. Distance:", d.min())
            print("  Mean Distance:", d.mean())
            print("  Median Distance:", np.median(d))
            print("  Max. Distance:", d.max())
    else:
        if verbose:
            print("Not enough sites to compute LD decay.")
        r2 = np.nan
        d = np.nan

    return r2, d


# ================================================================= #
# Storage
#
# ================================================================= #


def store_genetics(ks, hap, pos, ac,
                   og,
                   t0, div_samp_ct,
                   nsnps, track_ibd):
    """
    Store genetic statistics
    """
    # Compute
    k_stats = calc_k_stats(ks)
    barcode_counts = get_barcodes(hap)
    div_stats = calc_diversity_stats(pos, ac)
    
    # Store
    og.loc[div_samp_ct]["t0"] = t0
    # K
    for key, val in list(k_stats.items()):
        og.loc[div_samp_ct][key] = val
    # Diversity
    for key, val in list(div_stats.items()):
        og.loc[div_samp_ct][key] = val
    # Barcode
    og.loc[div_samp_ct]['n_barcodes'] = len(barcode_counts)
    og.loc[div_samp_ct]['single_barcodes'] = (barcode_counts == 1).sum()
    # IBD
    if track_ibd:
        frac_ibd, n_ibd, l_ibd = calc_ibd_statistics(hap)
        og.loc[div_samp_ct]['avg_frac_ibd'] = frac_ibd.mean()
        og.loc[div_samp_ct]['avg_n_ibd'] = n_ibd.mean()
        og.loc[div_samp_ct]['avg_l_ibd'] = l_ibd.mean()
    
    return og


def store_sfs(ks, ac,
              t0_sfs, binned_sfs_array,
              t0, div_samp_ct,
              bins):
    """
    Store Binned SFS array
    """
    
    # Compute unfolded SFS
    n = ks.sum()
    unfolded_sfs = calc_unfolded_sfs(ac, n=n)
    
    # Bin
    try:
        binned_sfs, _, _ = scipy.stats.binned_statistic(x=np.arange(1, n) / float(n),
                                                        values=unfolded_sfs,
                                                        statistic='sum',
                                                        bins=bins)
    except ValueError: # raised if n <= 2
        binned_sfs = np.nan
        
    # Store
    t0_sfs[div_samp_ct] = t0
    binned_sfs_array[div_samp_ct] = binned_sfs
        
    return t0_sfs, binned_sfs_array


def store_r2(ks, hap, pos, ac,
             t0_r2, binned_r2_array,
             t0, div_samp_ct,
             bins):
    """
    Stored binned r2 array
    """
    
    # Compute r2
    r2, d = calc_r2_decay(hap, pos, ac)
    
    # Bin
    try:
        binned_r2, _, _ = scipy.stats.binned_statistic(x=d,
                                                       values=r2,
                                                       statistic='mean',
                                                       bins=bins)
    except ValueError:
        binned_r2 = np.nan
    
    # Store
    t0_r2[div_samp_ct] = t0
    binned_r2_array[div_samp_ct] = binned_r2
        
    return t0_r2, binned_r2_array
