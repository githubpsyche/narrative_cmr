# -*- coding: utf-8 -*-
# %% [markdown]
# ## Simulation Configuration
# Let's demonstrate how to efficiently simulate the Landscape Model using the stimuli from our SBS dataset.

# %% [markdown]
# ### Dependencies and Data

# %%
import pandas as pd
import numpy as np
import json
from psifr import fr

# load collection of similarity matrices and convert to numpy
with open('data/similarities.json', 'r') as f:
    connections = json.load(f)
for key in connections.keys():
    connections[key] = np.array(connections[key])

# load recall data frame
data = pd.read_csv('data/psifr_sbs.csv')
events = fr.merge_free_recall(
    data, list_keys=['item_index', 'cycle', 'story_index', 
                     'story_name', 'time_test'])
events.head()

# %% [markdown]
# ### Cycle Extraction
# Let's just directly get a list of cycles containing indices of relevant units.

# %%
cycle_table = events.pivot_table(index=['story_name'], columns='input', values='cycle')
cycle_table

# %%
experiences = {}

for story_name in connections.keys():
    v = cycle_table.loc[story_name].values
    
    next_experience = []
    current_cycle = 0
    experiences[story_name] = []

    for unit_index, cycle_index in enumerate(v):
        if current_cycle != cycle_index:
            experiences[story_name].append(next_experience)
            next_experience = [unit_index]
            current_cycle = cycle_index
        else:
            next_experience.append(unit_index)

# %% [markdown]
# ### Recall Rates
# We're going to be testing the correlation between connection strengths and unit recall rates for each story, across subjects.

# %%
recall_rates = events.loc[(events.time_test==2)].pivot_table(index='input', columns='story_name', values='recall')
recall_rates.head()

# %% [markdown]
# ### Simulation

# %%
connection_strengths = {}
distance_ranks = {}

for story_name in connections.keys():
    connection_strengths[story_name] = []
    distance_ranks[story_name] = []

for story_name in connections.keys():
    model = LandscapeRevised(connections[story_name])
    model.connections[np.eye(model.unit_count, dtype='bool')] = 0
    
    # track connection strengths for each unit
    connection_strengths[story_name].append(np.sum(model.connections, axis=0))
    
    # track distance_rank score across units
    distance_rank = fr.distance_rank(
        events.loc[(events.story_name==story_name) & (events.time_test==1)], 
        'item_index', 1-model.connections).agg(['mean'])
    distance_ranks[story_name].append(distance_rank)


# %%
from scipy import stats

all_recall_rates = []
all_connection_strengths = []
for story_name in connections.keys():
    print(story_name)
    
    for i in range(len(connection_strengths[story_name])):
        story_recall_rates = recall_rates[story_name].values
        story_recall_rates = story_recall_rates[np.logical_not(np.isnan(story_recall_rates))]
        all_recall_rates = all_recall_rates + story_recall_rates.tolist()

        all_connection_strengths = all_connection_strengths + connection_strengths[story_name][i].tolist()
    
stats.pearsonr(all_connection_strengths, all_recall_rates)

# %%
connections

# %% [markdown]
# ## Exploration

# %% [markdown]
# We need some testing code to help confirm whether the model actually works or not as we develop it. What is our metric? Developing our `Semantic_Effects` notebook would clarify this. 
#
# In the 6/15 report, I specify a few analyses. One is model fits, another is clustering by representational similarity, another is a vague collection of semantic and temporal organizational analyses. The semantic analyses I had in mind were the lag-rank analyses supported by the psifr toolbox. The clustering analyses were also lag-rank analyses, I guess, but using representations simulated using the Landscape Model! And then the almost simpler analyses demoed in the Yeari et al (2016) and Cutler et al (2019) papers, where we measure the correlation between the connectivity between idea units (measured as the mean or sum of similarities between each idea unit and every other idea unit) against the recall rates of each idea unit.
#
# Let's try to build a visualization that starts with the baseline connection weights but also tracks statistics across each update of the Landscape model. Since even the very first version of the model will have those initial weights, this gives me a clean way to scale up and do that all within the constrained objectives of this notebook.
#
# This is doing what? Initialize the model with relevant connections - also presumably loading relevant data and figuring out how to encode individual cycles. Then at the end of each cycle simulation, I want to do the lag_rank analysis, getting a scalar representing the extent to which recall is clustered on the basis of my distance metric.
#
# Do this for each story. Plot a line for each story representing the timecourse of each score as the representation evolves.
#
# At the end of all simulations, plot for both initial and simulated connections the correlation between summed connectivity and unit recall rate.

# %%

# %% [markdown]
# ## Notes

# %% [markdown]
# ### How do we want to multiply activations and sigma?

# %% [markdown]
# It's multiplying each connection value in row i with each corresponding activation value j, then summing over column. Is that better?
#
# >  The activation spread from a connected unit j to a target unit i equals the multiplication of the activation of j from the previous cycle (Ajc−1) with the connection strength between units i and j as computed in the previous cycle (Sijc−1)
#
# Each index in activation corresponds to to the activation of j from the previous cycle. The connection strength between unit i and j as computed in the previous cycle is at index (i, j). So the right values are being accessed? Well, what is being summed over? Before we compute our sum, we are tracking for each connected unit j its activation (a[j]) times its connection with unit i (c[i, j]). The summation of this vector (collapsing over all connected units) reflects the activation spread to target unit i. So it's not 0, 22, 76, 162 that we want here, but 14, 38 62, 86.
#
# So yeah, we just want the dot product.

# %%
connections = np.arange(16).reshape(4, 4)
activations = np.arange(4)

activations, connections

# %%
connections @ activations

# %%
activations @ connections

# %% [markdown]
# Since our connections matrix is always symmetric, in practice we'd see the same result for either operation order.

# %% [markdown]
# ### How Many Parameters Does This Model _Really_ Have?

# %% [markdown]
# Outside of CMR, it's sort of 5? We list:
#
# - connections: array, initial connection strengths between text units  
# - max_activity: maximum activation that units are allowed to have  
# - min_activity: minimum activation that units are allowed to have  
# - decay_rate: decay rate of unit activation from one cycle to next  
# - memory_capacity: total activation possible in any given cycle  
# - learning_rate: rate of connection weight changes across cycles  
# - semantic_strength: relative contribution of initial semantic  
#     connections to computation of overall connection strengths  
#     
# Connections are static within a DSM (we could individualize this in a different project). `min_activity` should technically always be 0 according to the specification in Yeari et al. THe rest really do depend on individual differences according to the model. 
#
# If we decide to integrate the model into CMR, some parameters seem like they should correspond with some parameters in CMR. CMR has its own learning rate parameters, its own drift rate parameters, and its own scalaing of semantic association strengths. These of course have different consequences, and to use the same parameters would be to propose some shared representational substrate. An interesting project might be to explore the extent to which parameters "correlate" between subjects.

# %% [markdown]
# ### Default Parameters

# %% [markdown]
# > The default values were chosen on the basis of theoretical grounds (e.g., working memory capacity is five times the maximal value of the text units; Kintsch, 1988), neutrality (e.g., λ is close to 1, the semantic strength coefficient equals 1), and computational constraints (e.g., δ is set to .1 to prevent text units from noncurrent cycles from reaching the maximal activation value).
#
# > All simulations used the same default values set by the model in the following parameters: (1) semantic strength coefficient = 1, (2) maximal activation = 1, (3) working memory capacity = 5, (4) learning rate = .9, and (5) activation decay rate = .1.

# %%
