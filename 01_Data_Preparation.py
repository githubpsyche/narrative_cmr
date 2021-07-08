# %% [markdown]
# # Data Preparation
# Most of our analyses don't work directly on the text data we preprocess above. Instead, we want something formatted more like the traditional object of free recall modeling: sequences of study events and recall events related based on item identity, serial position, and other task features. 
#
# We'll follow the approach of the [Psifr library](https://psifr.readthedocs.io/en/latest/index.html) and represent most of our data in a long table format, with each row corresponding to a study or recall event and tracking for each event a subject index, a trial index, an input or output position, and item id. To identify items, we'll just use the text of the corresponding source unit either studied or recalled. 
#
# Uniquely with respect to narrative recall data, we're also interested in tracking cues in narratives that connect items semantically. Given our understanding of how the Landscape and CMR models work, we'll focus on tracking:
#
# 1. Co-occurrence of idea units within the same sentence (characterized as reading "cycles" in the documentation of the Landscape model)
# 2. Semantic similarity between idea units as tracked in sentence embeddings corresponding to the Sentence-BERT vector space model of word semantics
#
# Cycle identities will be included within our long table representations of the data, but semantic similarity matrices between source units for each story will be tracked separately, retrieved when relevant for analyses based on event details.

# %% [markdown]
# ## Dataset Overview

# %%
#from IPython.display import Markdown

#def render_tex(tex_path, bib_path, csl_path):
#    result = !pandoc -C --ascii {tex_path} -f latex -t markdown_mmd --bibliography {bib_path} --csl {csl_path}
#    return Markdown('\n'.join(result))

#render_tex('writing/BrownSchmidt_Dataset.tex', 'writing/references.bib', 'writing/main/apa.csl')

# %% [markdown]
# Human raters have gotten us most of what we want in the spreadsheet at `data/raw/Narrative Recall Data.xlsx`. Most preprocessing using external data is devoted to identifying otherwise ambiguous relationships between source idea units.

# %%
# dependencies
import os
import spacy
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

# model for computing sentence embeddings
embedding_model = SentenceTransformer("paraphrase-MiniLM-L12-v2")

# model for detecting reading cycles
nlp = spacy.load("en_core_web_trf")

# key paths
source_directory = os.path.join('data', 'raw')
text_directory = os.path.join('data', 'texts')
target_directory = os.path.join('data', 'sequences', 'human')

# names for relevant passages
passage_names = ['Fisherman', 'Supermarket', 'Flight', 'Cat', 'Fog', 'Beach']

# we use the original xlsx
data = pd.read_excel(os.path.join(
    source_directory, 'Narrative Recall Data.xlsx'), 
                     list(range(22)), engine='openpyxl')

data[0].head()

# %% [markdown]
# ## Story Information
# - Strings identifying idea units within each story
# - Semantic similarity matrix between source idea units
# - Cycles grouping source idea units based on co-occurence in the same sentence
#
# Later when we process specific trials, we'll retrieve this information to identify study events in our final table.

# %%
all_cycles = []
all_source_units = []
all_similarities = []
story_sequence = []

for trial_index, trial in tqdm(data[0].groupby(['story', 'timeTest'])):
    
    # we only consider each story once
    if trial['timeTest'].values[0] > 1:
        continue
    
    # identify story
    story_index = trial['story'].values[0]
    story_sequence.append(story_index)
    
    # source units are reproduced perfectly in xlsx file
    source_units = [each for each in list(trial['origText']) if type(each) == str]
    
    # collect relevant text
    with open(os.path.join(
        text_directory, passage_names[story_index-1] + '.txt'), encoding='utf8') as f:
        story_text = f.read()
        
    # sort units into cycles based on co-occurence in the same sentence
    # build cycle vector assigning a cycle index to each idea unit
    cycles = []
    cycle_index = 0
    last = 0
    story_doc = nlp(story_text)
    
    for unit in source_units:
        
        # locate the unit in story_text
        unit_loc = story_text.index(unit)
        
        # find the sentence corresponding to its first character
        unit_sentence = story_doc.char_span(unit_loc, unit_loc+len(unit.strip())).sent.start
        
        # if the sentence differs from the last considered one, that's a new cycle
        if unit_sentence != last:
            cycle_index += 1
            last = unit_sentence

        cycles.append(cycle_index)
                
    # track semantic similarity between each source unit
    embeddings = embedding_model.encode(source_units)
    similarities = util.pytorch_cos_sim(embeddings, embeddings).detach().tolist()
    
    all_cycles.append(cycles)
    all_similarities.append(similarities)
    all_source_units.append(source_units)

# %% [markdown]
# Let's do a sanity check: lengths of cycle, similarity, and source unit vectors should be the same.

# %%
for i in range(len(all_source_units)):
    print(len(all_cycles[i]), len(all_similarities[i]), len(all_source_units[i]))

# %% [markdown]
# ## Trial Information

# %%
results = []

# consider each unique trial
for subject_index, subject in enumerate(data):
    for trial_index, trial in enumerate(
        data[subject].groupby(['story', 'timeTest'])):
        
        # identify story, timeTest (we already have subject_index)
        story_index = trial[0][0]-1
        timeTest = trial[0][1]
        passage_name = passage_names[story_index]
        
        # build study event list based on extracted story information
        for unit_index, unit in enumerate(all_source_units[story_index]):
            results.append(
                [subject, trial_index, 'study', unit_index+1, 
                 unit, unit_index, all_cycles[story_index][unit_index], 
                 story_index, passage_name, timeTest])
        
        # we only care about the posRec column
        # create a recall event wherever a value is stored
        for serialPos, posRec in enumerate(list(trial[1]['posRec'])):
            
            # move to next entry if value can't be cast as integer
            try:
                posRec = int(posRec)
            except ValueError:
                continue
            
            results.append(
                [subject, trial_index, 'recall', posRec,
                 all_source_units[story_index][serialPos-1], serialPos-1,
                 all_cycles[story_index][serialPos-1], 
                 story_index, passage_name, timeTest])
            
results = pd.DataFrame(results, columns=[
    'subject', 'list', 'trial_type', 'position', 'item', 'item_index', 'cycle', 
    'story_index', 'story_name', 'time_test'])

results.head()

# %% [markdown]
# ## Store Results

# %%
import json

# similarities
similarity_result = {passage_names[i]: all_similarities[i] 
                     for i in range(len(all_similarities))}

with open('data/similarities.json', 'w') as f:
    f.write(json.dumps(similarity_result))
    
results.to_csv('data/psifr_sbs.csv', index=False)
