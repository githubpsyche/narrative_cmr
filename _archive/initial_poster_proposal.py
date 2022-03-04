# %% [markdown]
# # <center> Comprehension and Retrieved Context: Dual Constraints on the Organization of Free Narrative Recall

# %% [markdown]
# ## Poster Outline

# %% [markdown]
# ### Abstract
# The big high-level points that I'd like to make with this project are that:
# 1. **Review Literature and Present Evidence:** Semantic organization in sequence memory is mediated by representations that evolve dynamically over the course of encoding just like temporal organization is (but models so far haven't accounted for this).
# 2. **Present Evidence:** Insights from discourse-level models of reading comprehension can be integrated into retrieved context accounts of sequence memory to account for the effect of both processes on offline recall simultaneously.
# 3. **Suggestion:** Deeper integration of insights from existing semantic processing work into context-based models of retrieval would likely further improve accounts of known effects and support new predictions 

# %% [markdown]
# ### Background
# Short literature review to set up to review (Semantic)CMR and Landscape models, and explain how I integrate them into LandscapeCMR. Note the persistence question emphasized in the Morton & Polyn analyses suggesting processes are distinct, but focus on what's probably new to audience: this comprehension model. Model diagram for CMR outlines evolutionary relationship of items to different contextual states using a labeled matrix where each row corresponds to a contextual state associated with a given item (see sketch). Model diagram for baseline semantic association model sketches a subset of a network of pre-existing associations between items - either in a representational similarity matrix or more abstractedly as visualization of differently weighted links between labeled nodes. Diagram for landscape model represents the evolution of these associations with arrow and overlay of different versions of this visualization with diminishing transparency. To show that both dynamics contribute to recall, arrows from both the temporal context and semantic panels are drawn to a labeled recall sequence. 

# %% [markdown]
# ### Materials and Method

# %% [markdown]
# ![](cutler_method.png)
#
# Something like this.

# %% [markdown]
# ### Model Fits
# One of these will definitely be a distribution of model fits. We can consider vanilla CMR (no semantic associations), semantic CMR (no evolving semantic relationships), Landscape Model (just item-based cueing), and LandscapeCMR (encode landscape model connections into semantic CMR). We'll do a raincloud plot, treating each subject as a unique data point. Text will explain fitting method (differential evolution on fitting accuracy). For LandscapeCMR, we'll use default parameters to avoid confounding the comparison by flexibility.

# %% [markdown]
# ### Semantic and Temporal Organizational Analyses
# We'll compare predicted (over all subjects) and actual SPC and CRP for a selection of models, as well as metrics of semantic organization a la Figure 5 of Morton & Polyn (2016). 
#
# We'll also reproduce variation of analysis from Cutler et al (2019) and Yeari et al (2016) relating unit centrality to recall probability, contrasting baseline and simulated measures thereof. Yeari et al found that the Landscape model's simulated connection strengths were positively associated with recall proportions (r s = .70, p < .01; Fig. 1b). Cutler et al identified a similar result for semantically congruent idea units at delayed recall.

# %% [markdown]
# ### Clustering by Representational Similarity
# Previous work has applied a distance rank analysis to summarize with a single scalar value the tendency to group together nearby items using various distance metrics, including serial order and semantic similarity. This analysis is also probably applicable to measure the extent how recall is clustered according to latent representational states inferred with our models. For example, the distance_rank analysis can be applied to data using semantic similarities from GloVe, but also to semantic connections simulated with the Landscape model.
#
# To really underline how dynamics within the Landscape model progressively _evolve_ a representation of semantic associations between items, we can simulate the study phase of each trial using the model's default parameters and track this distance_rank statistic at each increment. A horizontal line records the initial value based on pre-existing semantic associations, also the default matrix used for SemanticCMR and associated analyses.

# %% [markdown]
# ## Current Project State
# - All data and analyses are already prepared, including SPC, CRP, Sem-CRP, Lag-Rank, Distance-Rank analyses, thanks to `psifr` library. 
# - Landscape and Semantic CMR models and associated simulation functions are fully implemented. SemanticCMR is configured to optionally initialize with Landscape model connectivity matrix during either simulation or fitting.
# - Both have already updated to support the predictive framework for evaluating models of semantic organization in free recall outlined by Morton & Polyn (2016), generating item recall probabilities conditional on prior recall events. 
# - Loss and fitting functions are fully implemented for SemanticCMR, but not yet the Landscape or integrated model.
# - Relevant literature already broadly reviewed thanks to previous projects

# %%
