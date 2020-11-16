import numpy as np
import allel
import scipy.stats
import configparser


# ================================================================= #
# Genome Evolution
#
# ================================================================= #


def meiosis(quantum, nsnps, p_oocysts=0.5, bp_per_cM=20):
    """
    Run P.falciparum meiosis on the `quantum` of 
    strains that enter the vector, with...
    
    - OPTIONALLY: Number of oocysts drawn from min[10, Geo(p_o)]
    - or 1 oocyst
    - Random pairings from `quantum` to produce zygotes
    - Number of cross-overs max[1, Poi(bp_per_cM)]
    - Random pairings from `bivalent` for cross-overs
    - Recomb. rate uniform along chromosome
    
    ...and returning *all* progeny generated across
    *all* oocysts.
    
    Parameters
        quantum : ndarray, shape (k, nsnps)
            Array of parasite genomes which will
            undergo meiosis. If `k` is < 2,
            no recombination occurs.
        nsnps : int
            Number of SNPs per parasite genome.
        p_oocysts : float in 0 < p <= 1
            Number of oocysts is drawn from
            ~Geo(p_o), up to a maximum of 10. Note 
            that if p = 1, only a single oocyst
            is drawn every time.
        bp_per_cM : float
            Recombination rate. Default of results in an
            average of one CO per bivalent when
            `nsnps`=1000.

    Returns
        progeny: ndarray, shape (n_oocysts*4, nsnps)
            Array of parasite genomes, after they have
            undergone meiosis.
    
    """
    
    if len(quantum) > 1:
        
        # Compute cross-over rate *per bivalent*
        mean_n_co = 2 * nsnps / (bp_per_cM * 100)
        
        # Draw no. oocysts
        n_oocysts = np.min([np.random.geometric(p_oocysts), 10])
        
        # Pair strains to create zygotes, 1 per oocyst
        i = np.random.choice(len(quantum), n_oocysts*2)
        zygotes = list(zip(i[:-1:2], i[1::2]))
        
        # Run meiosis for each zygote
        progeny = []
        for zygote in zygotes:
            parentals = np.copy(quantum[zygote, :])
            
            if not (parentals[0] == parentals[1]).all(): # if identical, no need
                # Create bivalent
                bivalent = np.zeros((2, nsnps, 2), dtype='int8')
                bivalent[:, :, 0] = np.copy(parentals)
                bivalent[:, :, 1] = np.copy(parentals)

                # Prepare crossover events
                n_co = np.max([1, np.random.poisson(mean_n_co)])  # enforce at least 1 CO
                co_brks = np.random.choice(nsnps, n_co)
                co_brks.sort()
                i = np.random.choice(2, n_co*2)
                co_pairs = list(zip(i[:-1:2], i[1::2]))

                # Resolve crossovers
                for brk, pair in zip(co_brks, co_pairs):
                    bivalent[[0, 1], :brk, pair] = bivalent[[1, 0], :brk, pair[::-1]]

                # Segregate & store progeny
                progeny.append(np.vstack([bivalent[:, :, 1], bivalent[:, :, 0]]))
            else:
                progeny.append(np.vstack([parentals, parentals]))
        
        # Combine progeny across oocysts
        progeny = np.vstack(progeny)
    
    else:
        progeny = quantum
    
    return progeny


def minimal_meiosis(quantum, nsnps, bp_per_cM=20):
    """
    Put two parasite genomes of `nsnps`
    length through meiosis
    
    The meiosis code is largely lifted from
    `pf-meiosis`, though the implementation
    below is slightly faster as it avoids
    function call overhead.
    
    Note that, presently, only *two* strains 
    in the `quantum` of transmission undergo 
    meiosis; if `n` strains are in the quantum, 
    two will undergo meiosis and `n - 2` will 
    remain the same.
    
    Note the quantum array is modified here inplace.
    
    DEPRECIATED -- `simulation.py` now runs
    `meiosis()`, see above.
    
    Parameters
        quantum : ndarray, shape (k, nsnps)
            Array of parasite genomes from which two will
            be chosen to recombine. If `k` is < 2,
            no recombination occurs.
        bp_per_cM : int
            Recombination rate. Default of results in an
            average of one CO per bivalent when
            `nsnps`=1000.
        nsnps : int
            Number of SNPs per parasite genome.

    Returns
        quantum: ndarry, shape (k, nsnps)
            Array of parasite genomes, after recombination.
    """
    Morgans_per_chr = float(nsnps)/(bp_per_cM*100)
    mean_n_co = 2*Morgans_per_chr  # this is per bivalent
    
    if len(quantum) > 1:
        ii = np.random.choice(a=len(quantum), size=2, replace=False)
        parentals = quantum[ii]
        if not (parentals[0] == parentals[1]).all():
            # Create bivalent
            bivalent = np.zeros((2, nsnps, 2), dtype='int8')
            bivalent[:, :, 0] = np.copy(parentals)
            bivalent[:, :, 1] = np.copy(parentals)
            # Prepare crossover events
            n_co = np.max([1, np.random.poisson(mean_n_co)])  # enforce at least 1 CO
            co_brks = np.random.choice(nsnps, n_co)
            co_brks.sort()
            co_pairs = [np.random.choice([0,1], 2, replace=True).tolist() for _ in range(n_co)]
            # Resolve crossovers
            for brk, pair in zip(co_brks, co_pairs):
                bivalent[[0, 1], :brk, pair] = bivalent[[1, 0], :brk, pair[::-1]]
            # Segregate progeny
            progeny = np.vstack([bivalent[:, :, 1], bivalent[:, :, 0]])
            # Select progeny
            quantum[ii] = progeny[np.random.choice(4, 2, replace=False)]
    return quantum


# ================================================================= #
# Parasite Population Evolution
#
# ================================================================= #


def evolve_host(hh, ti, theta=0.0, drift_rate=0.0, nsnps=0, back_mutation=False):
    """
    Evolve the parasite genomes of a host forward `ti` days,
    simulating drift and mutation, and optionally
    allowing for back mutation
    
    Implements a continuous-time Moran model with
    mutation

    Parameters
        hh: ndarray, shape (npv, nsnps)
            Array containing parasite genomes for single host.
        ti: float|
            Amount of time to simulate forward, measured in days.
            I.e. difference between present time and last update.
        theta: float
            Mutation probability per base per drift event.
        drift_rate: float
            Expected number of drift events per day.
        nsnps: int
            Number of SNPs per parasite genome.
        back_mutation: bool
            Allow for back mutation?
    Returns
        hh: ndarray, shape (npv, nsnps)
            Array containing parasite genomes for a single genome,
            after evolving.
    """
    
    nreps = np.random.poisson(ti * drift_rate)  # number of reproductions
    
    if nreps > 0:
        # For each reproduction...
        selected = np.random.choice(len(hh), size=(nreps, 2))  # select strains
        mutations = np.random.uniform(size=nreps) < theta * nsnps  # determine if mutation
        n_mutations = mutations.sum()
        mutation_position = np.random.choice(nsnps, n_mutations)  # select mutation position
        mutation_allele = np.random.uniform(0, 1, n_mutations)  # generate mutation allele
        
        # Sequentially simulate reproductions
        j = 0
        for i, mutation in enumerate(mutations):
            if mutation:
                hh[selected[i, 0], mutation_position[j]] = mutation_allele[j]
                j += 1
            else:  # drift
                hh[selected[i, 1]] = hh[selected[i, 0]]
        
    return hh


def evolve_vector(vv, ti, theta=0.0, drift_rate=0.0, nsnps=0, back_mutation=False):
    """
    Evolve the parasite genomes of a vector forward `ti` days,
    simulating drift and mutation, and optionally
    allowing for back mutation

    Parameters
        vv: ndarray, shape (npv, nsnps)
            Array containing parasite genomes for single vector.
        ti: float
            Amount of time to simulate forward, measured in days.
            I.e. difference between present time and last update.
        theta: float
            Mutation probability per base per drift event.
        drift_rate: float
            Expected number of drift events per day.
        nsnps: int
            Number of SNPs per parasite genome.
        back_mutation: bool
            Allow back mutation?
    Returns
        vv: ndarray, shape (npv, nsnps)
            Array containing parasite genomes for a single genome,
            after evolving.
    """

    nreps = np.random.poisson(ti * drift_rate)  # number of reproductions
    
    if nreps > 0:
        # For each reproduction...
        selected = np.random.choice(len(vv), size=(nreps, 2))  # select strains
        mutations = np.random.uniform(size=nreps) < theta * nsnps  # determine if mutation
        n_mutations = mutations.sum()
        mutation_position = np.random.choice(nsnps, n_mutations)  # select mutation position
        mutation_allele = np.random.uniform(0, 1, n_mutations)  # generate mutation allele
        
        # Sequentially simulate reproductions
        j = 0
        for i, mutation in enumerate(mutations):
            if mutation:
                vv[selected[i, 0], mutation_position[j]] = mutation_allele[j]
                j += 1
            else:  # drift
                vv[selected[i, 1]] = vv[selected[i, 0]]
        
    return vv


def evolve_all_hosts(h, h_a, tis, drift_rate=0.0, theta=0.0, nsnps=0, back_mutation=False):
    """
    Run `evolve_hosts()` on every infected host in `h_a`
    
    See `evolve_hosts` for details.
    
    """
    for idh in np.arange(h_a.shape[0]):
        if h[idh] == 1:  # contains infection
            h_a[idh] = evolve_host(hh=h_a[idh], ti=tis[idh],
                       drift_rate=drift_rate, theta=theta,
                       nsnps=nsnps, back_mutation=back_mutation)
    return h_a
    
    
def evolve_all_vectors(v, v_a, tis, drift_rate=0.0, theta=0.0, nsnps=0, back_mutation=False):
    """
    Run `evolve_vectors()` on every infected vector in `v_a`
    
    See `evolve_vectors` for details.
    
    """
    for idv in np.arange(v_a.shape[0]):
        if v[idv] == 1:  # contains infection
            v_a[idv] = evolve_vector(vv=v_a[idv], ti=tis[idv],
                       drift_rate=drift_rate, theta=theta,
                       nsnps=nsnps, back_mutation=back_mutation)
    return v_a


# ================================================================= #
# Parasite Transmission
#
# ================================================================= #


def infect_host(hh, vv, p_k=0.1):
    """
    Infect a host with parasites from a vector

    A host is bitten by a vector causing `k` parasite
    genomes to be transferred to the host. On average
    ~1 genome is transferred, and ~half are replaced.


    Parameters
        hh: ndarray, shape (nph, nsnps)
            Array containing parasite genomes for single host.
        vv: ndarray, shape (npv, nsnps)
            Array containing parasite genomes for single vector.
        p_k: float
            Probability of transmission per parasite genome.
        recombination: bool
            Simulate recombination?

    Returns:
        new: ndarray, shape (npv, nsnps)
            Array containing new parasite genomes for host,
            after it has been bitten by `vv`.
    """

    k = np.max((1, np.random.binomial(len(vv), p_k)))  # number to transfer
    quantum = vv[np.random.choice(len(vv), k, replace=False)]  # which to transfer

    if hh.sum() == 0:  # host is uninfected
        rel_wts = np.zeros(k)
        rel_wts[:] = 1.0 / k
        pool = quantum
    else:
        rel_wts = np.zeros(k + len(hh))
        rel_wts[:k] = 0.5 / k
        rel_wts[k:] = 0.5 / len(hh)
        pool = np.concatenate((quantum, hh), axis=0)

    new = pool[np.random.choice(len(pool), len(hh), replace=True, p=rel_wts)]
    return new


def infect_vector(hh, vv, nsnps, p_k=0.1, p_oocysts=0.5, bp_per_cM=20):
    """
    Infect a vector with parasites from a host,
    with parasites undergoing meiosis

    Parameters
        hh: ndarray, shape (nph, nsnps)
            Array containing parasite genomes for single host.
        vv: ndarray, shape (npv, nsnps)
            Array containing parasite genomes for single vector.
        nsnps: int
            Number of SNPs per parasite genome.
        p_k: float
            Probability of transmission per parasite genome.
        p_o : float
            Number of oocysts is drawn from
            ~Geo(p_o), up to a maximum of 10.
        bp_per_cM : float
            Recombination rate. Default of results in an
            average of one CO per bivalent when
            `nsnps`=1000.

    Returns:
        new: ndarray, shape (npv, nsnps)
            Array containing new parasite genomes for vector,
            after it has bitten `hh`.

    """
    
    # Randomly choose `k` strains to transmit to vector
    k = np.max((1, np.random.binomial(len(hh), p_k)))
    quantum = hh[np.random.choice(len(hh), k, replace=False)]
    
    # Pass through meiosis
    meiotic_progeny = meiosis(quantum, nsnps, p_oocysts, bp_per_cM)
    m = len(meiotic_progeny)

    # Establish new infection
    if vv.sum() == 0:
        rel_wts = np.zeros(m)
        rel_wts[:] = 1.0 / m
        pool = meiotic_progeny
    else:
        rel_wts = np.zeros(m + len(vv))
        rel_wts[:m] = 0.5 / m
        rel_wts[m:] = 0.5 / len(vv)
        pool = np.concatenate((meiotic_progeny, vv), axis=0)

    new = pool[np.random.choice(len(pool), len(vv), replace=True, p=rel_wts)]
    return new


def minimal_infect_vector(hh, vv, nsnps, p_k=0.1, recombination=True):
    """
    Infect a vector with parasites from a host, with optional
    recombination.

    If `recombination=True`, recombination occurs prior to replication
    in the vector (i.e. prior to filling of the `len(vv)` slots).
    
    
    DEPRECIATED -- `simulation.py` now runs `infect_vector()`

    Parameters
        hh: ndarray, shape (nph, nsnps)
            Array containing parasite genomes for single host.
        vv: ndarray, shape (npv, nsnps)
            Array containing parasite genomes for single vector.
        nsnps: int
            Number of SNPs per parasite genome.
        p_k: float
            Probability of transmission per parasite genome.
        recombination: bool
            Simulate recombination?

    Returns:
        new: ndarray, shape (npv, nsnps)
            Array containing new parasite genomes for vector,
            after it has bitten `hh`.

    
    Improvements:
    - Desirable to have mean # transmitted = 1, yet with np.max(),
    mean is actually above 1. But, if 1 is minimum that can be trans-
    ferred, and we allow >1, then mean can never be 1.
    - This is still not perfect, say k = 3; only one of the oocysts
    will contain recombinant progeny
    """

    k = np.max((1, np.random.binomial(10, p_k)))
    quantum = hh[np.random.choice(len(hh), k, replace=False)]
    if recombination:
        quantum = minimal_meiosis(quantum, nsnps)

    if vv.sum() == 0:
        rel_wts = np.zeros(k)
        rel_wts[:] = 1.0 / k
        pool = quantum
    else:
        rel_wts = np.zeros(k + len(vv))
        rel_wts[:k] = 0.5 / k
        rel_wts[k:] = 0.5 / len(vv)
        pool = np.concatenate((quantum, vv), axis=0)

    new = pool[np.random.choice(len(pool), len(vv), replace=True, p=rel_wts)]
    return new
