# AUTOGENERATED! DO NOT EDIT! File to edit: notebooks/model_variants/Semantic_CMR.ipynb (unless otherwise specified).

__all__ = ['LandscapeRevised', 'cmr_spec', 'Semantic_CMR']

# Cell

import numpy as np

class LandscapeRevised:
    """
    The landscape model of reading as revised by Yeari and van den Broek (1996).
    To encode a text into the model, it is initially segmented into text units
    (e.g., words or propositions) and reading cycles (e.g., clauses or
    sentences) depending on researcher preference. Similarly, semantic
    connections between all text units are also computed before model
    initialization. In the original specification of LS-R, semantic
    connections are computed using LSA, but we leave configuration of initial
    semantic connections separate from the model. This revised landscape model
    computes fluctuations in the _activation_ of text units and in the
    _interconnections_ established between them throughout reading.

    Basic Parameters:
    - connections: array, initial connection strengths between text units
    - max_activity: maximum activation that units are allowed to have
    - min_activity: minimum activation that units are allowed to have
    - decay_rate: decay rate of unit activation from one cycle to next
    - memory_capacity: total activation possible in any given cycle
    - learning_rate: rate of connection weight changes across cycles
    - semantic_strength: relative contribution of initial semantic
        connections to computation of overall connection strengths

    Retrieval Parameters (imported from CMR, only relevant for free recall):
    - stop_probability_scale
    - stop_probability_growth
    - choice_sensitivity

    Attributes:
    - activations: vector, current activation of each relevant text unit
    - connections: array, current connection strengths between text units
    """

    def __init__(self, connections, stop_probability_scale=1.0,
                 stop_probability_growth=1.0, choice_sensitivity=1.0,
                 max_activity=1.0, min_activity=0.0, decay_rate=0.1,
                 memory_capacity=5.0, learning_rate=0.9,
                 semantic_strength=1.0):

        # store initial parameters
        self.unit_count = len(connections)
        self.stop_probability_scale = stop_probability_scale
        self.stop_probability_growth = stop_probability_growth
        self.choice_sensitivity = choice_sensitivity
        self.max_activity = max_activity
        self.min_activity = min_activity
        self.decay_rate = decay_rate
        self.memory_capacity = memory_capacity
        self.learning_rate = learning_rate
        self.semantic_strength = semantic_strength

        # model architecture is set of activations and connections
        # across units diagonal of connections is 0 since we disallow
        # self-connections
        self.connections = connections * self.semantic_strength
        self.activations = np.zeros(self.unit_count) + self.min_activity

        # other variables to help track encoding/retrieval across trials
        self.recall_total = 0
        self.cycle_index = 0
        self.retrieving = False
        self.recall = np.zeros(self.unit_count)
        self.preretrieval_activations = self.activations

    def experience(self, cycles):
        """
        Updates activations and connections based on content of current
        reading cycle.

        Activations are updated as a function of three simulated mechanisms:
        1. attention: units of the current cycle are activated to the
            highest value
        2. working memory: units from prior cycles carry residual activation
            (following a decay rule)
        3. long-term memory: units from prior cycles are reactivated via
            connections with text units that are active in the current cycle.

        Episodic connections are added to and augment baseline connection
        strengths throughout the dynamic flow from one reading cycle to the
        next. They are formed between text units that are coactivated (due to
        any activation mechanism) in the same reading cycle. The strength of
        episodic connections is a function of the activation levels of the
        interconnected text units, and it accumulates with each concurrent
        activation (following a logarithmic learning rule).

        Argument:
        - cycles: vector of counts of units processed in each cycle
        """

        for cycle in cycles:
            self.update_activations(cycle)
            self.update_connections(self.activations)
        self.cycle_index += len(cycles)

    def update_activations(self, cycle):
        """
        Updates unit activations based on current reading cycle.

        1. Previous cycle activations decay by a parametrized amount
        toward a parametrized minimum value and spread to connected units.
        3. Regardless of outcome, the activations of units in the current
            cycle are set to the maximum allowed value.
        4. Finally activations are reduced proportionately based on
            memory_capacity.
        """

        # Previous cycle activations decay by a parametrized amount toward
        # some parametrized minimum value and spread to connected units.
        sigma = np.tanh(3 * (self.connections - 1)) + 1

        # activations of current cycle units set to maximum allowed value
        self.activations[cycle] = self.max_activity

        #previous_activations = self.activations.copy()
        #for i in range(self.unit_count):
        #    self.activations[i] = self.decay_rate * np.sum(sigma[i] * previous_activations)
        self.activations = self.decay_rate * np.sum(sigma * self.activations, axis=1)

        # activations of current cycle units set to maximum allowed value
        self.activations[cycle] = self.max_activity

        # activations of all units get set between min and max activity params
        self.activations = np.maximum(self.activations, self.min_activity)
        self.activations = np.minimum(self.activations, self.max_activity)

        #  if sum of activations exceeds capacity limit,
        # activations are reduced proportionally to attain the limit
        total_activation = np.sum(self.activations)
        if total_activation > self.memory_capacity:
            self.activations *=  self.memory_capacity / total_activation

    def update_connections(self, activations):
        """
        Updates model connection weights based on current unit activations.

        Connection strength is accumulated from one cycle to the next as a
        function of the activation levels of the connected units. The
        learning_rate parameter controls the rate of change, with a high value
        representing a higher rate of learning from previous textual
        information. Because learning_rate or activation values cannot be
        smaller than 0, the connection strength necessarily is above 0, and
        changes are incremental.
        """
        self.connections += self.learning_rate * np.outer(
            activations, activations)
        self.connections[np.eye(self.unit_count, dtype='bool')] = 1

    def outcome_probabilities(self):
        """
        Current unit recall probabilities given model state.
        """

        activation = np.power(self.activations, self.choice_sensitivity)
        probabilities = np.zeros((self.unit_count + 1))
        probabilities[0] = min(self.stop_probability_scale * np.exp(
            self.recall_total * self.stop_probability_growth), 1.0  - (
            (self.unit_count - self.recall_total) * 10e-7))

        for already_recalled_item in self.recall[:self.recall_total]:
            activation[int(already_recalled_item)] = 0
        probabilities[1:] = (
            1-probabilities[0]) * activation / np.sum(activation)

        return probabilities

    def free_recall(self, steps=None):

        # ensure retrieval information is reset
        if not self.retrieving:
            self.recall = np.zeros(self.unit_count)
            self.recall_total = 0
            self.preretrieval_activations = self.activations
            self.retrieving = True

        # we retrieve until termination if steps is left unspecified
        if steps is None:
            steps = self.unit_count - self.recall_total
        steps = self.recall_total + steps

        # at each recall attempt
        while self.recall_total < steps:

            # the current state of context is used as a retrieval cue to
            # attempt recall of a studied item compute outcome probabilities
            # and make choice based on distribution
            outcome_probabilities = self.outcome_probabilities()
            if np.any(outcome_probabilities[1:]):
                choice = np.sum(
                    np.cumsum(outcome_probabilities) < np.random.rand())
            else:
                choice = 0

            # resolve and maybe store outcome
            # we stop recall if no choice is made (0)
            if choice == 0:
                self.retrieving = False
                self.activations = self.preretrieval_activations
                break
            self.recall[self.recall_total] = choice - 1
            self.recall_total += 1
            self.update_activations([choice - 1])
        return self.recall[:self.recall_total]

    def force_recall(self, choice=None):

        # ensure retrieval information is reset
        if not self.retrieving:
            self.recall = np.zeros(self.unit_count)
            self.recall_total = 0
            self.preretrieval_activations = self.activations
            self.retrieving = True

        # resolve and maybe store outcome
        # we stop recall if no choice is made (0)
        if choice is None:
            pass
        elif choice > 0:
            self.recall[self.recall_total] = choice - 1
            self.recall_total += 1
            self.update_activations([choice - 1])
        else:
            self.retrieving = False
            self.activations = self.preretrieval_activations
        return self.recall[:self.recall_total]

# Cell
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
    ('context', float64[::1]),
    ('start_context_input', float64[::1]),
    ('preretrieval_context', float64[::1]),
    ('recall', int32[::1]),
    ('retrieving', boolean),
    ('recall_total', int32),
    ('primacy_weighting', float64[::1]),
    ('probabilities', float64[::1]),
    ('mfc', float64[:,::1]),
    ('mcf', float64[:,::1]),
    ('encoding_index', int32),
    ('items', float64[:,::1]),
    ('semantic_scale', float64),
    ('similarities', float64[:,::1])
]

# Cell

#@jitclass(cmr_spec)
class Semantic_CMR:

    def __init__(self, presentation_count, similarities, parameters):

        # store initial parameters
        item_count = len(similarities)
        self.item_count = item_count
        self.encoding_drift_rate = parameters['encoding_drift_rate']
        self.start_drift_rate = parameters['start_drift_rate']
        self.recall_drift_rate = parameters['recall_drift_rate']
        self.shared_support = parameters['shared_support']
        self.item_support = parameters['item_support']
        self.learning_rate = parameters['learning_rate']
        self.primacy_scale = parameters['primacy_scale']
        self.primacy_decay = parameters['primacy_decay']
        self.stop_probability_scale = parameters['stop_probability_scale']
        self.stop_probability_growth = parameters['stop_probability_growth']
        self.choice_sensitivity = parameters['choice_sensitivity']

        # specialized support for semantic connections when MCF is the cue
        self.semantic_scale = parameters['semantic_scale']
        self.similarities = np.vstack((np.zeros((1, item_count)), similarities, np.zeros((1, item_count))))

        # at the start of the list context is initialized with a state
        # orthogonal to the pre-experimental context
        # associated with the set of items
        self.context = np.zeros(item_count + 2)
        self.context[0] = 1
        self.preretrieval_context = self.context
        self.recall = np.zeros(item_count, dtype=np.int32) # recalls has at most `item_count` entries
        self.retrieving = False
        self.recall_total = 0

        # predefine primacy weighting vectors
        self.primacy_weighting = parameters['primacy_scale'] * np.exp(
            -parameters['primacy_decay'] * np.arange(presentation_count)) + 1

        # preallocate for outcome_probabilities
        self.probabilities = np.zeros((item_count + 1))

        # predefine contextual input vectors relevant for delay_drift_rate and start_drift_rate parameters
        self.start_context_input = np.zeros((self.item_count+2))
        self.start_context_input[0] = 1

        # The two layers communicate with one another through two sets of
        # associative connections represented by matrices Mfc and Mcf.
        # Pre-experimental Mfc is 1-learning_rate and pre-experimental Mcf is
        # item_support for i=j. For i!=j, Mcf is shared_support.
        self.mfc = np.eye(item_count, item_count+2, 1) * (1-self.learning_rate)
        self.mcf = np.ones((item_count, item_count)) * self.shared_support
        for i in range(item_count):
            self.mcf[i, i] = self.item_support
        self.mcf =  np.vstack((np.zeros((1, item_count)), self.mcf, np.zeros((1, item_count))))
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

    def update_context(self, drift_rate, experience):

        # first pre-experimental or initial context is retrieved
        if len(experience) == len(self.mfc):

            # if the context is not pre-experimental, the context is retrieved
            context_input = np.dot(experience, self.mfc)
            context_input = context_input / np.sqrt(
                np.sum(np.square(context_input))) # norm to length 1
        else:
            context_input = experience

        # updated context is sum of context and input, modulated by rho to have len 1 and some drift_rate
        rho = np.sqrt(1 + np.square(min(drift_rate, 1.0)) * (
            np.square(self.context * context_input) - 1)) - (
                min(drift_rate, 1.0) * (self.context * context_input))
        self.context = (rho * self.context) + (min(drift_rate, 1.0) * context_input)

    def activations(self, probe, use_mfc=False):

        if use_mfc:
            return np.dot(probe, self.mfc) + 10e-7
        elif self.semantic_scale == 0.0:
            return np.dot(probe, self.mcf) + 10e-7
        else:
            return (self.semantic_scale * np.dot(probe, self.similarities)
                   ) + np.dot(probe, self.mcf) + 10e-7

    def outcome_probabilities(self):

        self.probabilities[0] = min(self.stop_probability_scale * np.exp(
            self.recall_total * self.stop_probability_growth), 1.0 - (
                 (self.item_count-self.recall_total) * 10e-7))
        self.probabilities[1:] = 10e-7

        if self.probabilities[0] < (1.0 - ((self.item_count-self.recall_total) * 10e-7)):

            # measure the activation for each item
            activation = self.activations(self.context)

            # already recalled items have zero activation
            activation[self.recall[:self.recall_total]] = 0

            if np.sum(activation) > 0:

                # power sampling rule vs modified exponential sampling rule
                activation = np.power(activation, self.choice_sensitivity)

                # normalized result downweighted by stop prob is probability of choosing each item
                self.probabilities[1:] = (
                    1-self.probabilities[0]) * activation / np.sum(activation)

        return self.probabilities

    def free_recall(self, steps=None):

        # some amount of the pre-list context is reinstated before initiating recall
        if not self.retrieving:
            self.recall = np.zeros(self.item_count, dtype=np.int32)
            self.recall_total = 0
            self.preretrieval_context = self.context
            self.update_context(self.start_drift_rate, self.start_context_input)
            self.retrieving = True

        # number of items to retrieve is # of items left to recall if steps is unspecified
        if steps is None:
            steps = self.item_count - self.recall_total
        steps = self.recall_total + steps

        # at each recall attempt
        while self.recall_total < steps:

            # the current state of context is used as a retrieval cue to attempt recall of a studied item
            # compute outcome probabilities and make choice based on distribution
            outcome_probabilities = self.outcome_probabilities()
            if np.any(outcome_probabilities[1:]):
                choice = np.sum(np.cumsum(outcome_probabilities) < np.random.rand(), dtype=np.int32)
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
            self.recall = np.zeros(self.item_count, dtype=np.int32)
            self.recall_total = 0
            self.preretrieval_context = self.context
            self.update_context(self.start_drift_rate, self.start_context_input)
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