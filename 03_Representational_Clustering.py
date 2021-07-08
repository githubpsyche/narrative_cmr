# %% [markdown]
# # Clustering by Representational Similarity

# %% [markdown]
# Previous work has applied a distance rank analysis to summarize with a single scalar value the tendency to group together nearby items using various distance metrics, including serial order and semantic similarity. This analysis is also probably applicable to measure the extent how recall is clustered according to latent representational states inferred with our models. For example, the distance_rank analysis can be applied to data using semantic similarities from GloVe, but also to semantic connections simulated with the Landscape model.
#
# To really underline how dynamics within the Landscape model progressively _evolve_ a representation of semantic associations between items, we can simulate the study phase of each trial using the model's default parameters and track this distance_rank statistic at each increment. A horizontal line records the initial value based on pre-existing semantic associations, also the default matrix used for SemanticCMR and associated analyses.

# %% [markdown]
# ## Load Relevant Dependencies and Data
# For flexibility, we'll retrieve our own similarities.

# %%
from Landscape_Model import LandscapeRevised
from sentence_transformers import SentenceTransformer, util
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from psifr import fr
import spacy
import warnings
warnings.filterwarnings('ignore')

# load recall data frame
data = pd.read_csv('data/psifr_sbs.csv')
events = fr.merge_free_recall(
    data, list_keys=['item_index', 'cycle', 'story_index', 
                     'story_name', 'time_test'])

# paraphrase-MiniLM-L12-v2
# average_word_embeddings_glove.6B.300d
# average_word_embeddings_glove.840B.300d
# stsb-distilbert-base
model = SentenceTransformer('average_word_embeddings_glove.840B.300d')
units = events.pivot_table(index=['story_name', 'input'], values='item', aggfunc='first').reset_index()
connections = {}
remove_stopwords = False
nlp = spacy.load('en_core_web_sm')

for story_name in ['Fisherman', 'Supermarket', 'Flight', 'Cat', 'Fog', 'Beach']:
    
    sentences = units.loc[units.story_name==story_name].item.values.tolist()
    
    clean_sentences = []
    for i in range(len(sentences)):
        if remove_stopwords:
            text_token = nlp(sentences[i])
            clean_sentences.append(' '.join([word.text for word in text_token if not word.is_stop]))
        else:
            clean_sentences.append(sentences[i])
    
    #Compute embeddings
    embeddings = model.encode(clean_sentences, convert_to_tensor=True)

    #Compute cosine-similarities for each sentence with each other sentence
    cosine_scores = np.abs(util.pytorch_cos_sim(embeddings, embeddings).numpy())
    cosine_scores[np.eye(len(cosine_scores), dtype='bool')] = np.nan
    connections[story_name] = cosine_scores

events.head()

# %% [markdown]
# ## Demo Representational Clustering Analysis for Initial Model State
# For each story and time_test, initialize the model with the relevant connectivity matrix, perform the lag_rank analysis over the dataset using the matrix, combine dataFrames, and plot the result.

# %%
distance_ranks = []

# build list of distance_rank dfs across each factor i'm interested
for time_test in pd.unique(events.time_test):
    for story_name in pd.unique(events.story_name):

        # initialize the model with the relevant connectivity matrix
        model = LandscapeRevised(connections[story_name])
        model.connections[np.eye(model.unit_count, dtype='bool')] = 0

        # perform the distance_rank analysis over the dataset using the matrix
        distance_rank = fr.distance_rank(
            events.loc[(events.story_name==story_name) & (events.time_test==time_test)], 
            'item_index', 1-model.connections).reset_index()

        distance_rank['story_name'] = story_name
        distance_rank['time_test'] = time_test
        distance_ranks.append(distance_rank)
        
distance_rank = pd.concat(distance_ranks)
distance_rank = distance_rank.loc[distance_rank.time_test != 1]
distance_rank = distance_rank.pivot_table(index=['time_test', 'subject'], values='rank').reset_index()
distance_rank.head()

# %% [markdown]
# **Note**: Some of these rank values are nan for a given subject and condition. This is because participants didn't recall anything during these particular trials. This doesn't seem to affect downstream analyses. We'll demonstrate as much with our successive analysis: a dotplot of semantic organization scores factored by time_test and subject.

# %%
sns.set(style='whitegrid')
sns.lmplot(data=distance_rank, 
    x="time_test", y="rank", palette="deep");
plt.xticks([2, 3], ['Immediate', 'Delayed'])
plt.axhline(y=.5, color='r', label='chance')
plt.xlim([1.5, 3.5])
plt.ylim([.45, .65])
plt.xlabel('Time of Test')
plt.ylabel('Baseline Representational Clustering');

# %% [markdown]
# ## Simulation Configuration
# Before getting into more detailed analyses, let's demonstrate how to efficiently simulate the Landscape Model using the stimuli from our SBS dataset.

# %% [markdown]
# DataFrame construction should look much like the above, except with an extra factor varied over: `simulation_step`. From there, I'll apply another `pivot_table`, this time generalizing over subjects (or perhaps letting seaborn do that for me with its confidence interval support). The objective is a lineplot relating `simulation_step` with mean organization score across subjects.
#
# But what about simulation configuration? The key thing to work through is how to operate the `cycles` argument of `LandscapeRevised.experience`. The important thing is that each entry of `cycles` selects the right entries of `self.activations` to update when I assign `self.max_activity` within `LandscapeRevised.update_activations`. This probably just requires a list of indices per entry, right? Do I already have code for that?

# %% [markdown]
# ### Cycle Extraction
# Let's just directly get a list of cycles containing indices of relevant units.

# %%
experiences = {}

cycle_table = events.pivot_table(index=['story_name'], columns='input', values='cycle')

for story_name in connections.keys():
    v = cycle_table.loc[story_name].values
    v = v[~np.isnan(v)]
    
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

print(experiences['Fisherman'])

# %% [markdown]
# ## Extend Distance_Rank Analysis Over Each Simulation Step
# This time, we'll take the representational clustering analysis we demoed above and apply it over each simulation step, tracking changes in analysis result.

# %%
sim_distance_ranks = []
sim_connections = {}

# build list of distance_rank dfs across each factor i'm interested
for time_test in pd.unique(events.time_test):
    for story_name in pd.unique(events.story_name):

        # initialize model and store initial sim_distance_rank df
        model = LandscapeRevised(connections[story_name])
        model.connections[np.eye(model.unit_count, dtype='bool')] = 0

        # perform the distance_rank analysis over the dataset using the matrix
        sim_distance_rank = fr.distance_rank(
            events.loc[(events.story_name==story_name) & (events.time_test==time_test)], 
            'item_index', 1-model.connections).reset_index()

        # factor-specific information
        sim_distance_rank['story_name'] = story_name
        sim_distance_rank['time_test'] = time_test
        sim_distance_rank['simulation_step'] = 0
        sim_distance_ranks.append(sim_distance_rank)

        # add a further inner loop over cycles in story_name
        for cycle_index, cycle in enumerate(experiences[story_name]):
            model.experience([cycle])

            # perform the distance_rank analysis over the dataset using the matrix
            sim_distance_rank = fr.distance_rank(
                events.loc[(events.story_name==story_name) & (events.time_test==time_test)], 
                'item_index', 1-model.connections).reset_index()

            # factor-specific information
            sim_distance_rank['story_name'] = story_name
            sim_distance_rank['time_test'] = time_test
            sim_distance_rank['simulation_step'] = int(cycle_index + 1)
            sim_distance_ranks.append(sim_distance_rank)

        sim_connections[story_name] = model.connections.copy()

sim_distance_rank = pd.concat(sim_distance_ranks)
#sim_distance_rank = sim_distance_rank.loc[sim_distance_rank.time_test != 1]
sim_distance_rank.head()

# %% [markdown]
# Let's confirm that the analysis is solid by reproducing our above plot for just a single simulation_step in our data.

# %%
subset = sim_distance_rank[sim_distance_rank.simulation_step==10].pivot_table(index=['time_test', 'subject'], values='rank').reset_index()

sns.set(style='whitegrid')
sns.lmplot(data=subset, 
    x="time_test", y="rank", palette="deep");
plt.xticks([1, 2, 3], ['First Immediate', 'Second Immediate', 'Delayed'])
plt.axhline(y=.5, color='r', label='chance')
plt.xlim([.5, 3.5])
plt.xlabel('Time of Test')
plt.ylabel('Representational Clustering');

# %% [markdown]
# Next is a line plot relating simulation_step with representational clustering score.

# %%

sns.set(style='darkgrid')
g = sns.lineplot(data=sim_distance_rank, x='simulation_step', y='rank', hue='time_test', palette='pastel')
plt.xlabel('Simulation Step')
plt.ylabel('Representational Clustering in Recall')
plt.title('Clustering by Representational Similarity: Landscape Model')
plt.legend(['First Immediate', 'Second Immediate', 'Delayed'], title='Time of Test');

# %% [markdown]
# And again, but factored by story.

# %%
sns.set(style='darkgrid')

g = sns.FacetGrid(sim_distance_rank, 
    col='story_name', height=5)
g.map_dataframe(sns.lineplot, 'simulation_step', 'rank', hue='time_test', palette='pastel');
#g.set(xticks=np.arange(0, 46, 2))
g.set_xlabels('Simulation Step')
g.set_ylabels('Representational Clustering in Recall')
plt.show()

# %% [markdown]
# ## Semantic Similarity Matrix Follow-Up

# %%
for story_name in sim_connections.keys():
    print(story_name)
    print(np.nanmax(sim_connections[story_name]), np.nanmin(sim_connections[story_name]))
    print(np.median(sim_connections[story_name]), np.nanmax(sim_connections[story_name])/np.median(sim_connections[story_name]))

# %%
for story_name in sim_connections.keys():
    
    sns.heatmap(sim_connections[story_name], xticklabels=5, yticklabels=5, vmin=0, vmax=1)
    plt.title(story_name)
    plt.show()

# %% [markdown]
# Some cells in these connectivity matrices contain unbelievably high weights compared to other values, with the maximum cell value ranging from 70 times to 544 times the median cell value. This could be a sign of some bug in the model simulation.

# %% [markdown]
# ## Correlation Follow-Up
# Yeari et al found that the Landscape model's simulated connection strengths were positively associated with recall proportions (r s = .70, p < .01; Fig. 1b). Can we reproduce that finding here? We'd redo `Lmplot_Probability_Recall_by_Mean_Glove840B_Cosine_Similiarity.svg` from the `Cutler_Poster_Reproduction`, but using fully simulated Landscape Model representations instead of initial similarities.

# %%
sim_connection_strengths = {}
for story_name in sim_connections.keys():
    sim_connection_strengths[story_name] = np.nansum(sim_connections[story_name], axis=1)

strengths_df = events.pivot_table(
    index=['story_name', 'time_test', 'input'], values='recall').reset_index()
strengths_df['cosine_similarity'] = np.nan

for story_name in pd.unique(events.story_name):
    for time_test in range(1, 4):
        for input in range(1, len(sim_connection_strengths[story_name])+1):
            if len(strengths_df.loc[(strengths_df.story_name == story_name) & (
                strengths_df.time_test == time_test) & (strengths_df.input == input)]) == 1:

                strengths_df.loc[(strengths_df.story_name == story_name) & (
                    strengths_df.time_test == time_test) & (
                        strengths_df.input == input), 'cosine_similarity'] = sim_connection_strengths[story_name][input-1]

strengths_df.head()

# %%
sns.set(style='whitegrid')
g = sns.FacetGrid(strengths_df.loc[strengths_df.time_test == 1], 
    col='story_name', height=5)
g.map_dataframe(sns.lineplot, 'input', 'cosine_similarity');
g.set(xticks=np.arange(0, 46, 2))
plt.show()

# %%
sns.set_theme(style='whitegrid')
    
sns.lmplot(data=strengths_df.loc[strengths_df.time_test > 1], 
    x="cosine_similarity", y="recall", palette="deep", hue='time_test', legend=False);
plt.xlabel('connection strength')
plt.ylabel('probability recall');
plt.legend(['immediate', 'delay'], title='time of test');

# %% [markdown]
# Totally flat! Quite concerning that I couldn't reproduce the result. Yeah, I probably do have to look into this more closely.

# %% [markdown]
# ## Semantic CRP Follow-Up
# Similarly redo `FacetGrid_SemCRP_by_Time_Test` from the `Cutler_Poster_Reproduction` but using simulated model connectivities. 

# %%
sem_crps = []

# choose bins for CRP
bin_size = .1
np.arange(0, 1 + bin_size, bin_size)
edges = np.arange(0, 1 + bin_size, bin_size)

# build list of sem_crps across each factor i'm interested
for time_test in pd.unique(events.time_test):
    for story_name in pd.unique(events.story_name):
        subset = events.loc[(events.time_test == time_test) & (
            events.story_name == story_name)]
        dcrp = fr.distance_crp(
            subset, 'item_index', sim_connections[story_name], edges)
        dcrp['story_name'] = story_name
        if time_test == 1:
            dcrp['time_test'] = 1
        elif time_test == 2:
            dcrp['time_test'] = 'immediate'
        else:
            dcrp['time_test'] = 'delayed'
        sem_crps.append(dcrp)
    
sem_crp = pd.concat(sem_crps).reset_index()
sem_crp = sem_crp.loc[sem_crp.time_test != 1]
sem_crp = sem_crp.pivot_table(index=['time_test', 'subject', 'center'], values='prob').reset_index()
sem_crp

# %%
g = sns.FacetGrid(data=sem_crp, col='time_test')
g.map_dataframe(sns.lineplot, x='center', y='prob')
g.map_dataframe(sns.scatterplot, x='center', y='prob', hue='subject', palette='pastel')
g.set_xlabels('Similarity')
g.set_ylabels('CRP');

# %% [markdown]
#
