# %% [markdown]
# # The Revised Landscape Model
#
# Here we reproduce the revision of the Landscape model of reading comprehension presented by Yeari and van den Broek (2016). The model integrates the dynamic landscape model of reading comprehension originally characterized by van den Broek (1996) with a latent semantic analysis (LSA) representation of semantic knowledge. This revised landscape model (LS-R model) computes fluctuations in the activation of text units and the interconnections established between them throughout reading. Our implemention of the landscape model is, however, agnostic about the basis of representations of semantic knowledge.
#
# > Yeari, M., & van den Broek, P. (2016). A computational modeling of semantic knowledge in reading comprehension: Integrating the landscape model with latent semantic analysis. Behavior research methods, 48(3), 880-896.

# %% [markdown]
# ## The Model

# %%
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
        self.activations = self.decay_rate * np.sum(sigma * self.activations, axis=0)
        
        # activations of current cycle units set to maximum allowed value
        self.activations[cycle] = self.max_activity
        
        # activations of all units get set to at least minimum activation
        self.activations = np.maximum(self.activations, self.min_activity)
        
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
        self.connections[np.eye(self.unit_count, dtype='bool')] = 0
        
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


