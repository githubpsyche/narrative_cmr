# %% [markdown]
# # Semantic CMR

# %%

import numpy as np
from numba import float64, int32, boolean
from numba.experimental import jitclass

cmr_spec = [
    ('item_count', int32), 
    ('encoding_drift_rate', float64),
    ('start_drift_rate', float64),
    ('recall_drift_rate', float64),
    ('shared_support', float64),
    ('item_support', float64),
    ('learning_rate', float64),
    ('primacy_scale', float64),
    ('primacy_decay', float64),
    ('stop_probability_scale', float64),
    ('stop_probability_growth', float64),
    ('choice_sensitivity', float64),
    ('semantic_scale', float64),
    ('similarities', float64[:, ::1])
    ('context', float64[::1]),
    ('preretrieval_context', float64[::1]),
    ('recall', float64[::1]),
    ('retrieving', boolean),
    ('recall_total', int32),
    ('primacy_weighting', float64[::1]),
    ('mfc', float64[:,::1]),
    ('mcf', float64[:,::1]),
    ('encoding_index', int32),
    ('items', float64[:,::1]),
]


# %%

@jitclass(cmr_spec)
class SemanticCMR:

    def __init__(self, item_count, encoding_drift_rate, start_drift_rate, 
                 recall_drift_rate, shared_support, item_support, 
                 learning_rate, primacy_scale, primacy_decay, 
                 stop_probability_scale, stop_probability_growth, 
                 choice_sensitivity, semantic_scale, similarities):
        
        # store initial parameters
        self.item_count = item_count
        self.encoding_drift_rate = encoding_drift_rate
        self.start_drift_rate = start_drift_rate
        self.recall_drift_rate = recall_drift_rate
        self.shared_support = shared_support
        self.item_support = item_support
        self.learning_rate = learning_rate
        self.primacy_scale = primacy_scale
        self.primacy_decay = primacy_decay
        self.stop_probability_scale = stop_probability_scale
        self.stop_probability_growth = stop_probability_growth
        self.choice_sensitivity = choice_sensitivity
        self.semantic_scale = semantic_scale
        self.similarities = similarities
        
        # at the start of the list context is initialized with a state 
        # orthogonal to the pre-experimental context associated with items
        self.context = np.zeros(item_count + 1)
        self.context[0] = 1
        self.preretrieval_context = self.context
        self.recall = np.zeros(item_count)
        self.retrieving = False
        self.recall_total = 0

        # predefine primacy weighting vectors
        self.primacy_weighting = primacy_scale * np.exp(
            -primacy_decay * np.arange(item_count)) + 1

        # The two layers communicate with one another through two sets of 
        # associative connections represented by matrices Mfc and Mcf. Pre-
        # experimental Mfc is 1-learning_rate and pre-experimental Mcf is 
        # item_support for i=j. For i!=j, Mcf is shared_support.
        self.mfc = np.eye(item_count, item_count+1, 1) * (1 - learning_rate)
        self.mcf = np.ones((item_count, item_count)) * shared_support
        for i in range(item_count):
            self.mcf[i, i] = item_support
        self.mcf =  np.vstack((np.zeros((1, item_count)), self.mcf))
        self.encoding_index = 0
        self.items = np.eye(item_count, item_count)

    def experience(self, experiences):
        
        for i in range(len(experiences)):
            self.update_context(self.encoding_drift_rate, experiences[i])
            self.mfc += self.learning_rate * np.outer(
                self.context, experiences[i]).T
            self.mcf += self.primacy_weighting[self.encoding_index] * np.outer(
                self.context, experiences[i])
            self.encoding_index += 1

    def update_context(self, drift_rate, experience=None):

        # first pre-experimental or initial context is retrieved
        if experience is not None:
            context_input = np.dot(experience, self.mfc)
            context_input = context_input / np.sqrt(
                np.sum(np.square(context_input))) # norm to length 1
        else:
            context_input = np.zeros((self.item_count+1))
            context_input[0] = 1

        # updated context is sum of context and input, 
        # modulated by rho to have len 1 and some drift_rate
        rho = np.sqrt(1 + np.square(drift_rate) * (
            np.square(self.context * context_input) - 1)) - (
            drift_rate * (self.context * context_input))
        self.context = (rho * self.context) + (drift_rate * context_input)

    def activations(self, probe, use_mfc=False):

        if use_mfc:
            return np.dot(probe, self.mfc) + 10e-7
        elif self.semantic_scale == 1.0:
            return np.dot(probe, self.mcf) + 10e-7
        else:
            return (self.semantic_scale * np.dot(probe, self.similarities)
                   ) + np.dot(probe, self.mcf) + 10e-7
        
    def outcome_probabilities(self, activation_cue):

        activation = self.activations(activation_cue)
        activation = np.power(activation, self.choice_sensitivity)
        
        probabilities = np.zeros((self.item_count + 1))
        probabilities[0] = min(self.stop_probability_scale * np.exp(
            self.recall_total * self.stop_probability_growth), 1.0  - (
                (self.item_count-self.recall_total) * 10e-7))

        for already_recalled_item in self.recall[:self.recall_total]:
            activation[int(already_recalled_item)] = 0
        probabilities[1:] = (
            1-probabilities[0]) * activation / np.sum(activation)

        return probabilities

    def free_recall(self, steps=None):

        # some amount of the pre-list context is reinstated before recall
        if not self.retrieving:
            self.recall = np.zeros(self.item_count)
            self.recall_total = 0
            self.preretrieval_context = self.context
            self.update_context(self.start_drift_rate)
            self.retrieving = True

        # we retrieve until termination if steps is left unspecified
        if steps is None:
            steps = self.item_count - self.recall_total
        steps = self.recall_total + steps
        
        # at each recall attempt
        while self.recall_total < steps:

            # the current state of context is used as a retrieval cue to 
            # attempt recall of a studied item compute outcome probabilities 
            # and make choice based on distribution
            outcome_probabilities = self.outcome_probabilities(self.context)
            if np.any(outcome_probabilities[1:]):
                choice = np.sum(
                    np.cumsum(outcome_probabilities) < np.random.rand())
            else:
                choice = 0

            # resolve and maybe store outcome
            # we stop recall if no choice is made (0)
            if choice == 0:
                self.retrieving = False
                self.context = self.preretrieval_context
                break
            self.recall[self.recall_total] = choice - 1
            self.recall_total += 1
            self.update_context(self.recall_drift_rate, self.items[choice - 1])
        return self.recall[:self.recall_total]

    def force_recall(self, choice=None):

        if not self.retrieving:
            self.recall = np.zeros(self.item_count)
            self.recall_total = 0
            self.preretrieval_context = self.context
            self.update_context(self.start_drift_rate)
            self.retrieving = True

        if choice is None:
            pass
        elif choice > 0:
            self.recall[self.recall_total] = choice - 1
            self.recall_total += 1
            self.update_context(self.recall_drift_rate, self.items[choice - 1])
        else:
            self.retrieving = False
            self.context = self.preretrieval_context
        return self.recall[:self.recall_total]
