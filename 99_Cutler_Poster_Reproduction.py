# %% [markdown]
# # Cutler Poster Reproduction
# Let's confirm that our pipeline is sound by reproducing the main analyses from the Cutler (2019) poster on this same dataset.

# %% [markdown]
# ## Dependencies

# %%
import pandas as pd
import numpy as np
import json
import seaborn as sns
import matplotlib.pyplot as plt
from psifr import fr

# load recall data frame
data = pd.read_csv('data/psifr_sbs.csv')
events = fr.merge_free_recall(
    data, list_keys=['item_index', 'cycle', 'story_index', 
                     'story_name', 'time_test'])
events.head()

# %% [markdown]
# ## Recall Rates
# Her first figure compared recall rates between immediate and delayed recall conditions, plotting a unique point for each subject in the dataset. 

# %%
recall_rates_by_timetest = events.pivot_table(
    index=['subject', 'time_test'], values='recall').reset_index()
recall_rates_by_timetest.head()

# %% [markdown]
# I have to figure out which timeTest columns she used for each condition. My hypothesis based on her methodology panel is that she used the second column for her 'immediate' condition and the third column for her 'delayed' condition.

# %%
sns.set_theme(font_scale=1.3, style='darkgrid')

sns.catplot(
    x='time_test', y="recall", 
    data=recall_rates_by_timetest.loc[recall_rates_by_timetest.time_test > 1], 
        jitter=False, s=15, hue='subject', legend=False, palette='pastel');
plt.xticks(np.arange(2), ['immediate', 'delay'])
plt.yticks(np.arange(0, 1, .25), np.arange(0, 1, .25))
plt.xlabel('time of test')
plt.ylabel('probability recall')
plt.ylim([0, 1])
plt.show()

# %% [markdown]
# My hypothesis seems correct.

# %% [markdown]
# ## Serial Position Curve

# %% [markdown]
# She rescaled the data as percentiles to enable generalization over story length. I suspect this was a bad idea. I'll focus on producing this plot per story. 

# %%
spc = events.query('study').pivot_table(
    index=['subject', 'story_name', 'time_test', 'input'], values=['recall']).reset_index()
spc.reset_index(level=0, inplace=True)
spc = spc.loc[spc.time_test > 1]
spc.head()

# %%
sns.set(style='darkgrid')
g = sns.lineplot(data=spc, x='input', y='recall', hue='time_test', palette='pastel')
plt.xlabel('Study Position')
plt.ylabel('Probability Recall')
plt.legend(['immediate', 'delay'], title='time of test');

# %% [markdown]
# We _do_ find a steady decline in performance across serial position at immediate test, but not at delayed test.

# %% [markdown]
# ## Temporal Contiguity
# She compared lag-CRPs for the immediate and delayed conditions, across stories and subjects. We'll aggregate outputs from the `fr.lag_crp` function in `psifr`.

# %%
lag_crps = []
for time_test in pd.unique(events.time_test):
    lag_crps.append(fr.lag_crp(events[events.time_test == time_test]))
    
lag_crp = pd.concat(
    lag_crps, keys=pd.unique(events.time_test), names=['time_test']).reset_index()

lag_crp = lag_crp.loc[lag_crp.time_test > 1]
lag_crp.head()

# %%
sns.set_theme(font_scale=1.2, style="darkgrid")

max_lag = 10
filt_neg = f'{-max_lag} <= lag < 0'
filt_pos = f'0 < lag <= {max_lag}'

g = sns.FacetGrid(lag_crp, height=5)
g.map_dataframe(
    lambda data, **kws: sns.lineplot(
        data=data.query(filt_neg), x='lag', y='prob', hue='time_test', palette='pastel', **kws)
)
g.map_dataframe(
    lambda data, **kws: sns.lineplot(
        data=data.query(filt_pos), x='lag', y='prob', hue='time_test', palette='pastel', **kws)
)
g.set_xlabels('Lag')
g.set_ylabels('conditional response probability')
plt.legend(['immediate', 'delay'], title='time of test')
g.set(ylim=(0, 1));

# %% [markdown]
# ## Semantic Similarity Matrix
# First we need to apply a similarity metric close enough to that of the Cutler poster to reproduce the representational similarity matrices she visualized. She reports that she used the GloVe semantic vector space model and focused analysis on "content words", presumably excluding stop words as tracked in packages like `nltk` and `spacy`.

# %%
units = events.pivot_table(index=['story_name', 'input'], values='item', aggfunc='first').reset_index()
units.head()

# %%
from sentence_transformers import SentenceTransformer, util
import spacy

nlp = spacy.load('en_core_web_sm')
all_stopwords = nlp.Defaults.stop_words

# paraphrase-MiniLM-L12-v2
# average_word_embeddings_glove.6B.300d
# average_word_embeddings_glove.840B.300d
# stsb-distilbert-base
model = SentenceTransformer('average_word_embeddings_glove.840B.300d') 
connections = {}
remove_stopwords = False

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

# Let's take a peek at the ranges of these similarity scores.
# Technically, they should never be below 0, but norming is always an option.
for story_name in connections.keys():
    print(story_name)
    print(np.nanmax(connections[story_name]), np.nanmin(connections[story_name]))

# %%
for story_name in connections.keys():
    
    sns.heatmap(connections[story_name], xticklabels=5, yticklabels=5)
 #   print(np.nanmean(connections[story_name], axis=1))
    plt.title(story_name)
    plt.show()

# %% [markdown]
# The patterns in this matrix seem to be highly sensitive to the STS metric applied. And whether you filter for content words seems to have a meaningful impact, too.
#
# Eyeballing these plots suggests that the matrices in Becky's paper were from the Fog and Fisherman story facets, respectively. However, whether I remove stopwords or not, there doesn't see to be any one-to-one accord for any of these matrices.

# %% [markdown]
# ## Recall benefit for idea units that are more semantically similar to other idea units in the narrative
# It looks like for each idea unit across all stories, I need either a mean or summed similarity to every other relevant idea unit, along with its mean recall probability across subjects, split between time_test values. Plot the result and a line of best fit. Should see a nonsignificant positive relationship in the delayed condition but not the immediate.

# %%
connection_strengths = {}
for story_name in connections.keys():
    connection_strengths[story_name] = np.nanmean(connections[story_name], axis=1)

strengths_df = events.pivot_table(
    index=['story_name', 'time_test', 'input'], values='recall').reset_index()
strengths_df['cosine_similarity'] = np.nan

for story_name in pd.unique(events.story_name):
    for time_test in range(1, 4):
        for input in range(1, len(connection_strengths[story_name])+1):
            if len(strengths_df.loc[(strengths_df.story_name == story_name) & (
                strengths_df.time_test == time_test) & (strengths_df.input == input)]) == 1:

                strengths_df.loc[(strengths_df.story_name == story_name) & (
                    strengths_df.time_test == time_test) & (
                        strengths_df.input == input), 'cosine_similarity'] = connection_strengths[story_name][input-1]

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
plt.ylabel('probability recall');
plt.legend(['immediate', 'delay'], title='time of test');

# %% [markdown]
# ## Semantic CRP
# This wasn't in the original poster, but is a natural extension.
#
# The `psifr` library's `fr.distance_crp` function sorts distances into bins and then applies a lag_rank analysis to the result apparently. What's best practice for bin sorting? Cover the whole score range, size so a large enough sample is in each pool?

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
            subset, 'item_index', connections[story_name], edges)
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
plt.ylim([0, .2])


# %% [markdown]
# So the semantic CRP analysis is really noisy even when we include data from all stories. Standard deviation also seems to vary as semantic similarity increases, particularly in the delayed condition. That's the definition heteroskedasticity, right? I have no idea what the significance of that observation might be though. And the shape of this curve seems to depend substantially on the analysis anyway. I might end up concluding that I simply need more data. Will need some reflection about why though. In the meantime, let's clarify these analyses.

