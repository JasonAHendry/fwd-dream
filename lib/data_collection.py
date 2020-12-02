import random
import pandas as pd
import numpy as np
from numba import jit
import allel


# ================================================================= #
# Class to handle both prevalence and genetic data collection
# - Note the IBD related functions have been moved outside of the
#   class to allow for Numba optimisation
#
# ================================================================= #


class DataCollection(object):
    
    # Parasite prevalence statistics collected:
    prevalence_statistics = ["t0", "V1", "VX", "H1", "HX", "Hm", "HmX", "Vm", "VmX"]
    
    # Genetic diversity statistics collected:
    genetic_statistics = ["t0", "n_genomes", "n_samples", 
                          "n_mixed_samples", "frac_mixed_samples", "frac_mixed_samples",
                          "mean_k", "max_k", "n_singletons", "n_segregating",
                          "pi", "theta", "tajd"]
    
    # Optionally, IBS and IBD statistics collected:
    ibd_statistics = [prefix + suffix 
                      for prefix in ["f_ibd", "l_ibd", "f_ibs", "l_ibs"] 
                      for suffix in ["", "_bw", "_wn"]]
    
    
    # Initialise with no data
    def __init__(self, prev_samp_freq, div_samp_freq,
                 max_samples, detection_threshold, track_ibd):
        
        # Data Storage
        self.og = {k:[] for k in self.genetic_statistics}
        self.op = {k:[] for k in self.prevalence_statistics}
        
        self.track_ibd = track_ibd
        if self.track_ibd: 
            self.og.update({k:[] for k in self.ibd_statistics})
        
        # Frequency of collection
        self.tprev = 0
        self.prev_samp_freq = prev_samp_freq
        
        self.tdiv = 0
        self.div_samp_freq = div_samp_freq
        
        # Number of samples to collect
        self.max_samples = max_samples
        self.detection_threshold = detection_threshold

        
        
    """
    ================================================================================
    Collecting prevalence data
    --------------------------------------------------------------------------------
    
    """
    
    
    def sample_prevalence(self, t0, nh, nv, h1, v1, h_dt, v_dt):
        """
        Sample parasite prevalence for both hosts and vectors
        
        """
        # Compute
        sample_dt = {}
        sample_dt["t0"] = t0
        sample_dt["H1"] = h1
        sample_dt["V1"] = v1
        sample_dt["HX"] = h1 / nh
        sample_dt["VX"] = v1 / nv
        sample_dt["Hm"] = sum([self.detect_mixed(genomes, self.detection_threshold) 
                               for idh, genomes in h_dt.items()])
        sample_dt["Vm"] = sum([self.detect_mixed(genomes, self.detection_threshold) 
                               for idv, genomes in v_dt.items()])
        sample_dt["HmX"] = sample_dt["Hm"] / nh
        sample_dt["VmX"] = sample_dt["Vm"] / nh
        
        # Store
        for k, v in sample_dt.items():
            self.op[k].append(v)
            
        # Update time-of-last-sampling
        self.tprev = t0
            
        return sample_dt
    
        
    """
    ================================================================================
    Collecting genetic data
    --------------------------------------------------------------------------------
    
    """
    
    
    def sample_genetics(self, t0, h_dt):
        """
        Sample parasite genomes from the host population
        and compute a suite of genetic diversity statistics
        
        """
        # Collect genomes from hosts
        ks, genomes, ixs = self.collect_genomes(h_dt)
        
        # Get allele counts, define SNP positions
        ac = self.get_allele_counts(genomes)
        pos = np.arange(1, ac.shape[0]+1)
        
        # Compute statistics
        k_dt = self.calc_coi_statistics(ks)
        diversity_dt = self.calc_diversity_statistics(pos, ac)
        ibd_dt = dict(calc_ibd_statistics(genomes, ixs, rho=0.05, tau=0.1, theta=2.0))  # convert to dict from jit
        
        # Store
        sample_dt = {}
        sample_dt["t0"] = t0
        sample_dt.update(k_dt)
        sample_dt.update(diversity_dt)
        sample_dt.update(ibd_dt)
        for k, v in sample_dt.items():
            self.og[k].append(v)
            
        # Update time-of-last-sampling
        self.tdiv = t0
        
        return sample_dt

    
    """
    ================================================================================
    Sequencing DNA
    --------------------------------------------------------------------------------
    
    """
    
    
    def detect_mixed(self, oo, detection_threshold=None):
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
            nsnps = oo.shape[1]
            ans = n_het_sites / nsnps >= detection_threshold
        return ans
    
    
    def sequence_dna(self, hh, detection_threshold=None):
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
        
        
    def collect_genomes(self, h_dt):
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
        if self.max_samples is None or self.max_samples >= n_infected:
            n_collect = n_infected
        else:
            n_collect = self.max_samples

        # Collect and sequence DNA of samples
        ixs = []
        ks = []
        gs = []
        for idc in random.sample(h_dt.keys(), k=n_collect):
            k, g = self.sequence_dna(hh=h_dt[idc], detection_threshold=self.detection_threshold)
            ks.append(k)
            gs.append(g)
            ixs.extend([idc]*k)  # sample indices for each genome collected

        # Convert to numpy arrays
        genomes = np.vstack(gs).T  # transpose so rows=snps
        ks = np.array(ks)
        ixs = np.array(ixs)

        return ks, genomes, ixs
    
    
    """
    ================================================================================
    Calculating COI and genetic diversity statistics
    --------------------------------------------------------------------------------
    
    """
    
    
    def calc_coi_statistics(self, ks):
        """
        Calculate COI-related summary statistics
        given the COI distribution `ks`

        Parameters
            ks: ndarray, shape(n_samples)
                The COI distribution across collected
                samples.
        Returns
            k_dt: dict
                Dictionary containing COI summary statistics.
        """

        n_samples = len(ks)
        n_mixed_samples = (ks > 1).sum()

        k_dt = {"n_genomes": ks.sum(),                
                "n_samples": n_samples,
                "n_mixed_samples": n_mixed_samples,
                "frac_mixed_samples": n_mixed_samples / n_samples,
                "mean_k": ks.mean(),
                "max_k": ks.max()}

        return k_dt
    
    
    def get_allele_counts(self, genomes):
        """
        Generate an allele count array for a collection of genomes

        Parameters
            genomes: ndarray, shape (nsnps, n_genomes)
                Array encoding a set of sequenced parasite
                genomes.

        Returns:
            ac: AlleleCountArray, shape (nsnps, n_alleles)
                Allele counts for every loci in `genomes`.

        """
        nsnps, ngenomes = genomes.shape
        ac = np.zeros((nsnps, ngenomes), np.int16)  # the maximum possible size
        for i in np.arange(nsnps):
            counts = np.unique(genomes[i], return_counts=True)[1]
            n = len(counts)
            ac[i, :n] = counts
        ac = ac[:, ac.sum(0) > 0]  # remove columns with no alleles
        return allel.AlleleCountsArray(ac)
    
    
    def calc_diversity_statistics(self, pos, ac):
        """
        Calculate a suite of standard genetic diversity
        statistics given SNP positions `pos` and 
        allele counts `ac` for a set of parasite genomes

        Parameters
            ac: AlleleCountArray, shape (nsnps, nalleles)
                Allele counts for each position in the genome.
                
            pos: ndarray, int, shape (nsnps)
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
        else:
            n_segregating = ac.count_segregating()
            n_singletons = (ac == 1).any(1).sum()

        diversity_dt = {"n_segregating": n_segregating,
                        "n_singletons": n_singletons,
                        "pi": allel.stats.diversity.sequence_diversity(pos, ac),
                        "theta": allel.stats.diversity.watterson_theta(pos, ac),
                        "tajd": allel.stats.diversity.tajima_d(ac)}

        return diversity_dt
    
    

# ================================================================= #
# Calculating Identity-by-state (IBS) and Identity-by-descent (IBD)
#
# ================================================================= #


#@staticmethod
@jit(nopython=True)
def get_ibs_segments(ibs):
    """
    Given a boolean vector specifying whether
    each SNP position carries the same allele,
    return an array of identity track lengths

    Parameters
        ibs: ndarray, dtype bool, shape (nsnps)
            For each position in the genome, are
            the alleles identical?
    Returns
        segs: ndarray, dtype int, shape(n_segs)
            Return the length of IBS segments.
    """
    ll = []
    l = 0
    for state in ibs:
        if state:
            l += 1
        else:
            ll.append(l)
            l = 0
    ll.append(l)  # append last segment
    segs = np.array(ll)  # always positive, won't be >65K SNPs

    return segs[segs > 0]  # only return if positive length


#@staticmethod
@jit(nopython=True)
def calc_ibd_emissions(tau, theta):
    """
    Calculate the emission probabilities
    for the HMM inferring IBD from IBS
    in an infinite-sites model

    Here, IBD means coalescence has occurred
    before `tau` generations in the past; the
    site does not need to be identical.

    Parameters
        tau: float
            `tau` indicates the point in time
            after which all alleles are considered
            distinct, i.e. if two alleles have not
            coalesced before this time they are not
            in IBD. It is expressed in coalescent
            time units (N generations).
        theta: float
            The coalescent mutation rate.

    Returns
        emiss: ndarray, float, shape(2, 2)
            Emission probabilities for IBD
            and not IBD states. Rows are IBD state 
            and columns give probabilities of carrying 
            the same or different alleles.

    """

    # Eqns derived assuming neutral coalescent
    same_given_ibd = (1 / (1 + theta)) * (1 - np.exp(-(1 + theta)*tau))
    same_given_not_ibd = (1 / (1 + theta)) * np.exp(-(1 + theta)*tau)
    diff_given_ibd = (theta / (1 + theta)) - (np.exp(-tau) - np.exp(-tau * (1 + theta))/(1 + theta))
    diff_given_not_ibd = np.exp(-tau) - np.exp(-tau * (1 + theta))/(1 + theta)

    # Indexed such that 0=not IBD, 1=IBD
    emiss = np.array([[diff_given_not_ibd, same_given_not_ibd],  # not IBD
                      [diff_given_ibd, same_given_ibd]])  # IBD

    # Row probabilities must sum to one
    for i in range(2):
        emiss[i] /= emiss[i].sum()

    return emiss


#@staticmethod
@jit(nopython=True)
def get_ibd_segments(ibs, rho, emiss):
    """
    Given an IBS profile for two parasite genomes,
    return a set of IBD tracks calculated using
    a simple hidden Markov Model

    Parameters
        ibs_state: ndarray, bool, shape (nsnps)
            For each position in the genome report,
            True if both parasite genomes carry the
            same allele.
        rho: float
            The recombination rate between sites.
            Assumed uniform across the genome.
        emiss: ndarray, float64, shape(2, 2)
            Numpy array containing the emission probabilities
            for the HMM.

    Returns
        ibd_segs: ndarray, float32, shape(n_ibd_segments)
            Numpy array containing a list of the length
            of all the detected IBD segments.
    """

    # Parameters
    nsnps = ibs.shape[0]

    # Prepare data structures
    vt = emiss[ibs[0]]  # initialise
    tb = np.zeros((2, nsnps), np.int8)
    tb[1, 1:] = 1
    wmx = vt.argmax()
    vt /= vt[wmx]
    vtn = vt * (1 - rho)

    # Iterate
    for j in range(1, nsnps):
        for i in range(2):
            if vt[i] * (1 - rho) < rho:
                vtn[i] = rho
                tb[i, j] = wmx
            else:
                vtn[i] = vt[i] * (1 - rho)
            vtn[i] *= emiss[ibs[j], i]
        wmx = vtn.argmax()
        vt = vtn / vtn[wmx]

    # Traceback to produce IBD segment list
    state = vt.argmax()
    l = state
    ibd_segs = []
    for j in range(2, nsnps):
        state = tb[state, nsnps-j]
        if state:
            l += 1
        elif l > 0:
            ibd_segs.append(l)
            l = 0
    if l > 0: ibd_segs.append(l)  # append last segment

    return np.array(ibd_segs)


#@staticmethod
@jit(nopython=True)
def calc_ibd_statistics(genomes, ixs, rho, tau, theta):
    """
    Calculate IBD (and IBS) statistics for every pair 
    of `genomes` that have been collected, taking note
    of which sample indexes `ixs` they have come from
    to compute within- and between-sample IBD

    Parameters
        genomes: ndarray, float32, shape (nsnps, ngenomes)
            Array encoding a set of sequenced parasite
            genomes.
        ixs: ndarray, int, shape (ngenomes)
            Array containing the indexes of the samples
            each genome came from.
        rho: float
            The recomination rate.
        theta: float
            The coalescent mutation rate; classically
            equal to twice the product of the effective 
            population size and mutation rate.
        tau: float
            Indicates the time point after which all
            alleles are considered unique; two alleles
            are in IBD if they are copies of the same
            ancestral allele that existed before `tau`,
            i.e. MRCA existed before `tau`.


    Returns
        popn_dt: dict
            Dictionary containing IBD and IBS
            summary statisics for `genomes`.


    """
    nsnps, ngenomes = genomes.shape

    # Compute number of pairwise comparisons
    n_pairs = int(ngenomes * (ngenomes - 1) / 2)

    # Store
    pairwise_dt = {
        "f_ibs": np.zeros(n_pairs),
        "l_ibs": np.zeros(n_pairs),
        "f_ibd": np.zeros(n_pairs),
        "l_ibd": np.zeros(n_pairs),
    }
    pair_type = np.zeros(n_pairs, np.bool_)

    # Compute emission matrix for IBD HMM
    emiss = calc_ibd_emissions(tau, theta)

    # Perform pairwise comparisons
    ix = 0
    for i in range(ngenomes - 1):
        for j in range(i + 1, ngenomes):

            ibs = (genomes[:, i] == genomes[:, j]).astype(np.int8)
            ibs_segments = get_ibs_segments(ibs)
            ibd_segments = get_ibd_segments(ibs, rho, emiss)

            if len(ibs_segments) > 0:
                pairwise_dt["f_ibs"][ix] = ibs_segments.sum()
                pairwise_dt["l_ibs"][ix] = ibs_segments.mean()

            if len(ibd_segments) > 0:
                pairwise_dt["f_ibd"][ix] = ibd_segments.sum()
                pairwise_dt["l_ibd"][ix] = ibd_segments.mean()

            if ixs[i] == ixs[j]:  # within-sample comparison
                pair_type[ix] = 1

            ix += 1

    # Population-level summary statistics
    pairwise_dt["f_ibs"] /= nsnps
    pairwise_dt["f_ibd"] /= nsnps
    popn_dt = {}
    for k, v in pairwise_dt.items():
        popn_dt[k] = v.mean()
        popn_dt[k + "_wn"] = v[pair_type].mean() if pair_type.any() else 0
        popn_dt[k + "_bw"] = v[~pair_type].mean() if (~pair_type).any() else 0 

    return popn_dt
    
    