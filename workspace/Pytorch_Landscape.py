# %%
# hide
if 'google.colab' in str(get_ipython()):

    # mount google drive and move to directory
    #from google.colab import drive
    #drive.mount('/content/drive')
    # #%cd 'drive/MyDrive/landscape_cmr'

    # install local lbrary and nbdev
    # !pip3 install -e . -q
    # !pip3 install nbdev -q

    # notebook specific libraries
    # !pip install -U sentence-transformers

from nbdev.showdoc import *

# %% [markdown]
# # The Revised Landscape Model
#
# Here we reproduce the revision of the Landscape model of reading comprehension presented by Yeari and van den Broek (2016). The model integrates the dynamic landscape model of reading comprehension originally characterized by van den Broek (1996) with a latent semantic analysis (LSA) representation of semantic knowledge. This revised landscape model (LS-R model) computes fluctuations in the activation of text units and the interconnections established between them throughout reading. Our implemention of the landscape model is, however, agnostic about the basis of representations of semantic knowledge.
#
# > Yeari, M., & van den Broek, P. (2016). A computational modeling of semantic knowledge in reading comprehension: Integrating the landscape model with latent semantic analysis. Behavior research methods, 48(3), 880-896.
#

# %% [markdown]
# ## Setup

# %%
import torch
import numpy as np # for loading data
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

from sentence_transformers import SentenceTransformer
# from transformers import DistilBertModel, DistilBertTokenizer

# %%
password=input()
# !git clone https://spectraldoy:{password}@github.com/vucml/landscape_cmr

# %%
if torch.cuda.is_available():
  device = torch.device("cuda:0")
else:
  device = torch.device("cpu")


# %% [markdown]
# ## Semantic Text Similarity
# This should be as part of the data preprocessing, data can be saved as dictionaries with:
# ```python
# {
#   "text_units": torch.Tensor of text unit indices,
#   "init_connections": initial TextSimilarity between text units
# }
# ```

# %%
class TextSimilarity(nn.Module):
  def __init__(self, model_name="stsb-distilbert-base"):
    super(TextSimilarity, self).__init__()
    self.model_name = model_name

    # get the model and tokenizer, assuming we are using DistilBert
    self.model = SentenceTransformer(model_name)
    self.to(device)
  
  @staticmethod
  def cosine_similarity(x: torch.Tensor) -> torch.Tensor:
    """
    Custom implementation of cosine similarities
    """
    norm = x.norm(dim=-1).unsqueeze(0)

    # this is the formula for cosine similarities in a symmetric matrix
    return x @ x.t() / (norm.t() @ norm)

  def forward(self, cycles: list) -> torch.Tensor:
    """
    Assumes input is a list of lists of sentences/text units
    i.e. a List of Lists of strings
    """
    embeddings = torch.cat([self.model.encode(i, convert_to_tensor=True) for i in cycles])
    init_connections = self.cosine_similarity(embeddings)
    return init_connections


# %%
ts = TextSimilarity("stsb-distilbert-base")

# %%
reading_cycles = [
  ["this is one", "of the many", "we have", "to use"],
  ["this is another"],
  ["do you think", "this could be", "a final?"]
]
ts(reading_cycles)


# %% [markdown]
# ## The Model

# %%
class LandscapeRevised(nn.Module):
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
    
    Attributes:
    - activations: vector, current activation of each relevant text unit
    - connections: array, current connection strengths between text units
    """

    def __init__(self, connections, 
                 stop_probability_scale,
                 stop_probability_growth,
                 choice_sensitivity,
                 max_activity=1.0, 
                 min_activity=0.0, 
                 decay_rate=0.1, 
                 memory_capacity=5.0, 
                 learning_rate=0.9, 
                 semantic_strength=1.0):
        """
        Initializes model instance with the specified parameter configuration.
        Every relevant text unit comes with an activation level and set of 
        connection weights to every other text unit. Activation levels are 
        initialized to 0, while initial connection weights are specified by 
        the connections parameter. Other parameters regulate fluctuations in 
        unit activations and connection weights throughout reading.
        Parameters:  
        - connections: array, initial connection strengths between text units  
        - max_activity: maximum activation that units are allowed to have  
        - min_activity: minimum activation that units are allowed to have  
        - decay_rate: decay rate of unit activation from one cycle to next  
        - memory_capacity: total activation possible in any given cycle  
        - learning_rate: rate of connection weight changes across cycles  
        - semantic_strength: relative contribution of initial semantic 
            connections to computation of overall connection strengths  
        """
        super(LandscapeRevised, self).__init__()

        # set initial parameters
        self.max_activity = nn.Parameter(
            torch.tensor([max_activity], dtype=torch.float32, device=device)
        )
        self.min_activity = nn.Parameter(
            torch.tensor([min_activity], dtype=torch.float32, device=device)
        )
        self.decay_rate = nn.Parameter(
            torch.tensor([decay_rate], dtype=torch.float32, device=device)
        )
        self.memory_capacity = nn.Parameter(
            torch.tensor([memory_capacity], dtype=torch.float32, device=device)
        )
        self.learning_rate = nn.Parameter(
            torch.tensor([learning_rate], dtype=torch.float32, device=device)
        )
        self.semantic_strength = nn.Parameter(
            torch.tensor([semantic_strength], dtype=torch.float32, device=device)
        )

        # model architecture is set of activations and connections across units
        # diagonal of connections is nan since we disallow self-connections
        self.unit_count = len(connections)
        self.connections = connections * self.semantic_strength
        self.activations = torch.zeros(self.unit_count, device=device) + self.min_activity

        # retrieval parameters
        self.choice_sensitivity = choice_sensitivity
        self.stop_probability_scale = stop_probability_scale
        self.stop_probability_growth = stop_probability_growth

        # other variables to help track encoding and retrieval across trials
        self.recall_total = 0
        self.encoding_index = 0
        self.retrieving = False
        self.recall = torch.zeros(self.unit_count, device=device)
        self.preretrieval_activations = self.activations.clone()

    def reset(self, new_connections):
        """
        Reinitializes model state with the specified connections, but does not
        change the parameters, so that parameter learning can continue.
        
        Every relevant text unit comes with an activation level and set of 
        connection weights to every other text unit. Activation levels are 
        initialized to 0, while initial connection weights are specified by 
        the connections parameter. Other parameters regulate fluctuations in 
        unit activations and connection weights throughout reading.
        Parameters:  
        - new_connections: array, new initial connection strengths between text units
        """
        # model architecture is set of activations and connections across units
        # diagonal of connections is nan since we disallow self-connections
        self.unit_count = new_connections.shape[-1]
        self.connections = new_connections * self.semantic_strength
        self.activations = torch.zeros(self.unit_count, device=device) + self.min_activity

        # other variables to help track encoding and retrieval across trials
        self.recall_total = 0
        self.encoding_index = 0
        self.retrieving = False
        self.recall = torch.zeros(self.unit_count, device=device)
        self.preretrieval_activations = self.activations.clone()

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
            # correct cycle, assuming iterable
            for i in cycle:
              if i < 0 or i >= self.unit_count:
                raise IndexError(f"Index {i} in cycle is out of range")

            # experience
            self.update_activations(cycle) # removed += encoding index
            self.update_connections(self.activations)
            self.encoding_index += len(cycle)
        
    def update_activations(self, cycle):
        """
        Updates unit activations based on current reading cycle.
        1. The activation of each text unit in the current cycle is 
            computed as the sum of the activations spread from each connected 
            unit based on those units' activation in the previous cycle, 
            modulated by decay_rate.  
        2. Regardless of the outcome, the activations of units in the current 
            cycle are set to the maximum allowed value.  
        3. Finally activations are reduced proportionately based on 
            memory_capacity.  
        Argument:
        - cycle: vector of unit indices processed in this cycle
        """

        # spread of activations from previous cycle based on connection weights
        # with positive logarithmic change in connection strengths enforced
        sigma = torch.tanh(3 * (self.connections-1)) + 1
        sigma = sigma 
        self.activations = self.decay_rate * ( sigma @ self.activations.t() ).t().squeeze()

        # activations of current cycle units get set to maximum allowed value
        self.activations.index_put_([torch.cat(cycle)], self.max_activity)

        # activations of all units get set to at least minimum activation
        self.activations = torch.maximum(
            self.activations, torch.ones_like(self.activations, device=device) * self.min_activity
        )

        # total activations ensured less or equal to memory capacity
        total_activation = torch.sum(self.activations)
        if total_activation > self.memory_capacity:
            self.activations *= self.memory_capacity / total_activation


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

        self.connections += self.learning_rate * torch.outer(
            activations, activations)

    def outcome_probabilities(self):
        """
        Current unit recall probabilities given model state.
        """

        activations = torch.pow(self.activations, self.choice_sensitivity)
        probabilities = torch.zeros((self.unit_count + 1))
        probabilities[0] = min(self.stop_probability_scale * np.exp(
            self.recall_total * self.stop_probability_growth), 1.0)

        if probabilities[0] < 1:
            for unit in range(self.encoding_index):
                if unit in self.recall[:self.recall_total]:
                    continue
                probabilities[unit + 1] = self.activations[unit]
            probabilities[1:] *= (
                1 - probabilities[0]) / torch.sum(probabilities[1:])

        return probabilities

    def draft_probability(self):
        """
        Possible probability based on activations and connections
        """
        return torch.softmax(
            self.activations.unsqueeze(0).t() @ (self.connections @ self.activations.t()).unsqueeze(0),
            dim=-1
        )

    def force_recall(self, choice):
        """
        Forces model to recall chosen unit and updates model state.
        Here, recall items are 1-indexed, with a choice of 0 indicating a 
        choice to end retrieval and return to preretrieval model state.
        """
        if not self.retrieving:
            self.recall = torch.zeros(self.unit_count)
            self.recall_total = 0
            self.preretrieval_activations = self.activations
            self.retrieving = True

        if choice is None:
            pass
        elif choice == 0:
            self.retrieving = False
            self.activations = self.preretrieval_activations
        else:
            self.recall[self.recall_total] = choice - 1
            self.recall_total += 1
            self.update_activations(np.array([choice]))
        return self.recall[:self.recall_total]


    def free_recall(self, steps=None):
        """
        Simulates free recall for specified steps based on model state.
        """

        if not self.retrieving:
            self.recall = torch.zeros(self.unit_count)
            self.recall_total = 0
            self.preretrieval_activations = self.activations
            self.retrieving = True
            
        # number of items to retrieve is infinite if steps is unspecified
        if steps is None:
            steps = math.inf
        steps = self.recall_total + steps

        # at each recall attempt
        while self.recall_total < steps:

            # compute outcome probabilities, choose based on distribution
            outcome_probabilities = self.outcome_probabilities()
            if torch.any(outcome_probabilities[1:]):
                choice = torch.sum(
                    torch.cumsum(outcome_probabilities) < np.random.rand())
            else:
                choice = 0

            # resolve and maybe store outcome
            # we stop recall if no choice is made (0)
            if choice is None:
                pass
            elif choice == 0:
                self.retrieving = False
                self.activations = self.preretrieval_activations
            else:
                self.recall[self.recall_total] = choice - 1
                self.recall_total += 1
                self.update_activations(np.array([choice]))
        return self.recall[:self.recall_total]

# %%
init_connections = ts(reading_cycles)
lsr = LandscapeRevised(init_connections, 0.1, 0.1, 1)

# %%
lsr.experience([
                [ torch.tensor([5])],
                [ torch.tensor([6]),torch.tensor([7]) ]
                ])

# %%
lsr.activations

# %%
lsr.draft_probability()

# %%
F.cross_entropy(lsr.draft_probability(), torch.randperm(8, device=device))

# %% [markdown]
# ## Data Setup

# %%
reading_cycles1 = [
  ["this is one", "of the many", "we have", "to use"],
  ["this is another"],
  ["do you think", "this could be", "a final?"]
]
ic1 = ts(reading_cycles1)
cycle_idxs1 = []
count = 0
for i in range(len(reading_cycles1)):
  cycle_idxs1.append([])
  cycle_idxs1[i] = []
  for j in reading_cycles1[i]:
    cycle_idxs1[i].append(count)
    count += 1
cycle_idxs1

# %%
reading_cycles2 = [
  ["look at this graph", "said Nickel Back", "a long time ago"],
  ["or is it Nickel back", "or Nickle Back"],
  ["Perhaps"],
  ["we may never know", "said someone", "in an anonymous manner", "anonymously"]
]
ic2 = ts(reading_cycles2)
cycle_idxs2 = []
count = 0
for i in range(len(reading_cycles2)):
  cycle_idxs2.append([])
  cycle_idxs2[i] = []
  for j in reading_cycles2[i]:
    cycle_idxs2[i].append(count)
    count += 1
cycle_idxs2

# %%
data = [
  [torch.randperm(len(ic1), device=device), ic1, cycle_idxs1],
  [torch.randperm(len(ic2), device=device), ic2, cycle_idxs2]
]


# %%
class LandscapeDS(Dataset):
  def __init__(self, data):
    """
    data: a Python list of datapoints each containing
      - a np array of text unit indices in recall order (true value)
      - a matrix of initial connection weights
      - a list of lists denoting the reading cycles (convert to list of torch tensors?)
    """
    super(LandscapeDS, self).__init__()
    self.data = data
  
  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, item):
    return self.data[item]

# %%
ds = LandscapeDS(data)
dl = DataLoader(ds, batch_size=1, shuffle=True) # can't have >1 batch_size as size differs


# %% [markdown]
# # Training

# %%

# %%
# loss calculation
def loss_fn(model, datapoint, criterion=F.cross_entropy):
    model.reset(datapoint[1])
    model.experience(datapoint[2])

    return criterion(model.draft_probability(), datapoint[0])


# %%
def fit(epochs, model, opt, dl):
    """fit the model to the dl for the specified number of epochs"""
    losses = []
    print_interval = epochs // 10 + (epochs < 10)

    for epoch in range(epochs):
        # setup backprop
        model.train()
        loss = 0
        opt.zero_grad()

        # pass through the entire dataset
        for datapoint in dl:
            loss += loss_fn(model, datapoint)
        loss = loss / len(dl)
        
        # backprop
        loss.backward()
        opt.step()
        losses.append(float(loss))

        model.eval()
        # validation stuff

        if epoch % print_interval == 0:
          print(f"Epoch {epoch} Loss {loss}")

    return losses


# %%
# create model
item_zero = next(iter(dl))
lsr = LandscapeRevised(item_zero[1], 0.1, 0.1, 1).to(device)

# create optimizer
LR = 0.1 # needs to be tuned
optimizer = optim.SGD(lsr.parameters(), lr=LR)

# %%
lossses = fit(10, lsr, optimizer, dl)

# %%
lsr.state_dict()

# %%
# initial default params:
max_activity=1.0, 
min_activity=0.0, 
decay_rate=0.1, 
memory_capacity=5.0, 
learning_rate=0.9, 
semantic_strength=1.0
