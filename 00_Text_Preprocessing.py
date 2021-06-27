# -*- coding: utf-8 -*-
# %% [markdown]
# # Text Preprocessing

# %% [markdown]
# The raw data obtained from the Brown-Schmidt lab for this research requires extensive extra preprocessing to be suitable for downstream analyses. Here we do that by:
#
# - Removing irrelevant boilerplate
# - Putting text into a maintainable file format
# - Extracting serial order of study idea units from texts and researcher segmentations/correspondences
# - Pairing extracted trial data with semantic similarity data

# %% [markdown]
# ## Dataset Overview
# We render an overview of the dataset prepared for our publication:

# %%
from IPython.display import Markdown

def render_tex(tex_path, bib_path, csl_path):
    result = !pandoc -C --ascii {tex_path} -f latex -t markdown_mmd --bibliography {bib_path} --csl {csl_path}
    return Markdown('\n'.join(result))

render_tex('writing/BrownSchmidt_Dataset.tex', 'writing/references.bib', 'writing/main/apa.csl')

# %% [markdown]
# ## Standardizing Text Representations

# %% [markdown]
# Using the data in `raw`, we produce in `texts` one subdirectory for each passage (with passage contents at base) and in each subdirectory, one file for each recall period. Each file will contain only the recalled text associated with a particular passage, subject, and recall period and be labeled accordingly (e.g. as `Supermarket_1_1.txt`). At the base of `texts`, the text of the source passages will each be included as separate files.

# %% [markdown]
# ### We start with some initial dependencies and constants.

# %%
# import dependencies
import os
import pathlib
import docxpy
import ftfy

# key paths
source_directory = os.path.join('data', 'raw')
target_directory = os.path.join('data', 'texts')

source_names = ['Fisherman', 'Supermarket', 'Flight', 'Cat', 'Fog', 'Beach']
source_titles = ['where does susie go at noon?']
title_tags = [['''man and the bear'''], ['''act of kindness'''], 
              ["""a man can’t just sit""", "a man just can’t sit"], 
              ["where does susie go at noon?"], ["fog: a maine t"], 
              ["day at the beach"]]
author_tags = ['author unknown', 'anonymous', 'chris holm', 'adapted from',
               'unknown', 'anonymous']

# %% [markdown]
# ### Next we create directories in our file system to organize preprocessed data.

# %%
# make a pooled subdirectory if one doesn't already exist
if not os.path.isdir(target_directory):
    os.mkdir(target_directory)

# generate subdirectory for each passage
for source_name in source_names:
    passage_path = os.path.join(target_directory, source_name)
    if not os.path.isdir(passage_path):
        os.mkdir(passage_path)

# %% [markdown]
# ### Preprocess raw `docx` files and store as text

# %%
# for each pt1 written recall file, extract text and remove boilerplate, 
# and save to correct location in `pooled`
for path, subdirs, files in os.walk(os.path.join(
    source_directory, 'recall', 'Written Recall Part 1')):
    for name in files:
        recall_path = str(pathlib.PurePath(path, name))
        
        # extract text and remove boilerplate
        recall_text = '\n'.join(
            docxpy.process(recall_path).split('\n')[1:]).strip()
        passage_index = recall_path[-9:-8]
        subject_index = recall_path.split(name)[0][-3:-1]
        phase_index = recall_path[-7:-6]
        targetname = '{}_{}_{}.txt'.format(
            source_names[int(passage_index)-1], int(subject_index), phase_index)
        
        # handle special cases??
        recall_text = recall_text.replace(
            'vbeach', 'beach').replace('Susie gp at noon', 'Susie go at noon')
        
        # filter out source titles from recall data
        if any([each in recall_text[:recall_text.find(
            '.')].lower() for each in title_tags[int(passage_index)-1]]):
            if len(recall_text[:recall_text.find('\n')]) < 100:
                recall_text = recall_text[recall_text.find('\n'):].strip()
                
        # filter out source authors from recall data
        if (recall_text[:len(author_tags[int(
            passage_index)-1])].lower() == author_tags[int(passage_index)-1]):
            recall_text = recall_text[recall_text.find('\n'):].strip()
            
        # clean the data
        recall_text = ftfy.fix_text(recall_text)
            
        # save to correct location in pooled
        with open(
            os.path.join(target_directory, source_names[int(passage_index)-1], 
                         targetname), 'w', encoding='utf-8') as f:
            f.write(recall_text)

# %% [markdown]
# Part 1 and Part 2 data were collected in slightly different contexts, so they are preprocessed a little differently:

# %%
# for each pt2 written recall file, extract text and remove boilerplate, 
# and save to correct location in `pooled`
for path, subdirs, files in os.walk(
    os.path.join(source_directory, 'recall', 'Written Recall Part 2')):
    for name in files:
        recall_path = str(pathlib.PurePath(path, name))
        
        # identify correct location in pooled
        passage_index = recall_path[-7:-6]
        subject_index, phase_index =  recall_path.split(name)[0][-3:-1], 3
        if len(passage_index.strip()) == 0:
            continue
        targetname = '{}_{}_{}.txt'.format(
            source_names[int(passage_index)-1], int(subject_index), phase_index)

        # extract text and remove boilerplate
        boilerplate = 'You have 5 minutes to type the story you just read for memory. There is no word limit. Please write as much as you can remember.'
        recall_text = docxpy.process(
            recall_path).replace(boilerplate, '').strip()
        recall_text = '\n'.join(recall_text.split('\n')[1:]).strip()
        
        # clean text
        recall_text = ftfy.fix_text(recall_text)
        
        # save to correct location
        with open(os.path.join(
            target_directory, source_names[int(passage_index)-1], 
            targetname), 'w', encoding='utf-8') as f:
            f.write(recall_text)

# %% [markdown]
# ### The result is an organized directory of text representations of participant responses absent methodology-specific details such as the content of the recall prompt.
