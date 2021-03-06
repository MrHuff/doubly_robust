from abc import abstractmethod

import numpy as np
import scipy
from collections import defaultdict
import itertools
from policy_evaluation.Utils import *
import policy_evaluation.GammaDP as GammaDP
import joblib


# NonUniformGamma(...) computes a Gamma_pinv matrix for non-uniform exploration
# num_candidates: (int) Number of candidates, m
# decay: (double) Decay factor. Doc Selection Prob \propto exp2(-decay * floor[ log2(rank) ])
# ranking_size: (int) Size of slate, l
# allow_repetitions: (bool) If True, repetitions were allowed in the ranking
def NonUniformGamma(multinomial, n_items, n_reco):
    for i in range(1, n_items):
        prevVal = multinomial[i - 1]
        currVal = multinomial[i]
        if np.isclose(currVal, prevVal):
            multinomial[i] = prevVal

    gammaVals = GammaDP.GammaCalculator(multinomial.tolist(), n_reco)
    gamma = np.diag(np.ravel(gammaVals.unitMarginals))

    for p in range(n_reco):
        for q in range(p + 1, n_reco):
            pairMarginals = gammaVals.pairwiseMarginals[(p, q)]
            currentRowStart = p * n_items
            currentRowEnd = (p + 1) * n_items
            currentColumnStart = q * n_items
            currentColumnEnd = (q + 1) * n_items
            gamma[currentRowStart:currentRowEnd, currentColumnStart:currentColumnEnd] = pairMarginals
            gamma[currentColumnStart:currentColumnEnd, currentRowStart:currentRowEnd] = pairMarginals.T

    normalizer = np.sum(multinomial, dtype=np.longdouble)

    gammaInv = scipy.linalg.pinv(gamma)
    return gammaInv


"""
Classes represent policies that have recommend function mapping from context to recommendation(treatment)
"""


class Policy(object):
    def __init__(self, n_items, n_reco):
        """
        :param n_items: number of all items
        :param n_reco: number of recommendation
        """
        self.n_reco = n_reco
        self.n_items = n_items

    @abstractmethod
    def recommend(self, context):
        """
        recommend a permutation of items given a context
        :return: list of items
        """
        pass


"""
Sample items without replacement based on pre-define probability
"""


class MultinomialPolicy(Policy):
    def __init__(self, item_vectors, estimated_user_vectors, n_items, n_reco,
                 temperature=1.0, greedy=False, cal_gamma=False):
        """
        :param item_vectors: probability distribution over items
        :param greedy: if greedy is true -> recommend items which have the highest probabilities
        """
        super(MultinomialPolicy, self).__init__(n_items, n_reco)
        self.item_vectors = item_vectors
        self.estimated_user_vectors = estimated_user_vectors
        self.greedy = greedy
        self.tau = temperature
        self.multinomials = softmax(np.matmul(estimated_user_vectors, self.item_vectors.T),
                                    axis=1, tau=self.tau)
        self.gammas = None
        if cal_gamma:
            gammas = joblib.Parallel(n_jobs=-1, verbose=50)(
                joblib.delayed(NonUniformGamma)(m, n_items, n_reco) for m in self.multinomials)
            self.gammas = np.array(gammas)

    def get_propensity(self, multinomial, reco):
        """
        Calculate probability of given recommendation set
        """

        log_prob = 0.
        current_denom = multinomial.sum()
        for p in range(self.n_reco):
            log_prob += (np.log(multinomial[reco[p]]) - np.log(current_denom))
            current_denom -= multinomial[reco[p]]
            if current_denom <= 0:
                break

        return log_prob
        # prob = 1.0
        # current_denom = multinomial.sum()
        # for p in range(self.n_reco):
        #     prob *= (multinomial[reco[p]] / current_denom)
        #     current_denom -= multinomial[reco[p]]
        #     if current_denom <= 0:
        #         break
        #
        # return prob

    def recommend(self, user):
        user_vector = self.estimated_user_vectors[user]
        multinomial = self.multinomials[user]

        if self.greedy:
            reco = np.argsort(-multinomial, kind='mergesort')[:self.n_reco]
        else:
            reco = np.random.choice(len(multinomial), self.n_reco, p=multinomial, replace=False)
        return reco, multinomial, user_vector


"""
Sort items by popularity (number of clicks)
"""


class GlobalSortPolicy(Policy):
    def __init__(self, n_items, n_reco, sim_data):
        super(GlobalSortPolicy, self).__init__(n_items, n_reco)
        self.global_sort = self.get_mostpopular(sim_data)

    def get_mostpopular(self, sim_data):
        book_hotel = [d['h'] for d in sim_data if d['r'] > 0]
        hotel_booking = defaultdict(int)
        for h in book_hotel:
            hotel_booking[h] += 1
        return np.array(map(lambda x: x[0], sorted(hotel_booking.items(), key=lambda x: -x[1])))

    def recommend(self, context):
        return self.global_sort[:self.n_reco]


"""
Sort items by popularity given context(user)
"""


class MostCommonByUserPolicy(Policy):
    def __init__(self, n_items, n_reco, sim_data):
        super(MostCommonByUserPolicy, self).__init__(n_items, n_reco)
        self.sorting_map = self.get_mostpopular(sim_data)
        self.n_reco = n_reco
        self.n_items = n_items

    def get_mostpopular(self, sim_data):
        groupData = itertools.groupby(sorted(sim_data, key=lambda x: x['x']), key=lambda x: x['x'])
        for x, data in groupData:
            book_hotel = [d['h'] for d in data if d['r'] > 0]
            hotel_booking = defaultdict(int)
            for h in book_hotel:
                hotel_booking[h] += 1
        return np.array(list(map(lambda x: x[0], sorted(hotel_booking.items(), key=lambda x: -x[1]))))

    def recommend(self, context):
        return self.sorting_map[context][:self.n_reco]


"""
Random sort
"""


class RandomSortPolicy(Policy):
    def __init__(self, n_items, n_reco):
        super(RandomSortPolicy, self).__init__(n_items, n_reco)
        self.n_reco = n_reco
        self.n_items = n_items

    def recommend(self, context):
        return np.random.choice(self.n_items, self.n_reco, replace=False)


class FixedPolicy(object):
    def __init__(self, fixed_reco):
        self.fixed_reco = fixed_reco

    def recommend(self, x):
        return self.fixed_reco[x]
