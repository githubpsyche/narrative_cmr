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
# **Note**: Some of these rank values are nan for a given subject and condition. This is because participants didn't recall anything during these particular trials. Does this affect downstream analyses? We'll find out with our successive analysis: a dotplot of semantic organization scores factored by time_test and subject.

# %%

sns.set(style='whitegrid')
sns.lmplot(data=distance_rank, 
    x="time_test", y="rank", palette="deep");
plt.xticks([2, 3], ['immediate', 'delay'])
plt.axhline(y=.5, color='r', label='chance')
plt.xlim([1.5, 3.5])
plt.ylim([.45, .65])
plt.xlabel('time of test')
plt.ylabel('organization score');

# %% [markdown]
# ## Simulation Configuration
# Let's demonstrate how to efficiently simulate the Landscape Model using the stimuli from our SBS dataset.

# %%