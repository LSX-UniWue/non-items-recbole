from hyptrails.trial_roulette import *
def calc_evidence(matrix, trails, vocab, ax, label):

    shape_matrix = (max(vocab.values()) + 1, max(vocab.values()) + 1)
    
    # We iterate through some values of k (hypothesis weighting factor)
    evidences = {}
    for i in range(10):
        # If k = 0 then we can simply use an empty matrix
        if i == 0:
            prior = csr_matrix(shape_matrix, dtype=np.float64)
        # Otherwise, we need to build the hypothesis matrix and elicit the prior
        else:
            # Here we elicit the Dirichlet prior from expressed hypothesis matrix
            # We use the row-wise (trial) roulette chip distribution (i.e., same amount of chips for each row)
            # chips = number of chips to distribute PER ROW
            # The HypTrails paper suggests to distribute |S|*k chips per row.
            # As we ignore self-loops, we distribute (|S|-1)*k chips per row.
            chips = i*(matrix.shape[0]-1.)
            prior = distr_chips_row(matrix, chips, n_jobs=1)
            
        # Now, we can pass everything to the Markov chain framework and calculate corresponding evidences
        # k=1 corresponds to the order of the MC (first-order)
        # reset=False can be set to true if we want to work with a reset state (start end end state for each trail)
        # prior=1. corresponds to the initial uniform prior that is necessary for ensuring proper priors
        markov = MarkovChain(k=1, use_prior=True, reset = False, prior=1., specific_prior=prior,
                                specific_prior_vocab = vocab, modus="bayes")
        markov.prepare_data(trails)
        markov.fit(trails)

        evidence = markov.bayesian_evidence()
        evidences[i] = evidence
    
    ax.plot(evidences.keys(), evidences.values(), marker='o', clip_on = False, label=label, linestyle='--')
    return evidences