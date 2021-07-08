# %% [markdown]
# # <center> Relating Reading Comprehension with Retrieved Context Theory

# %% [markdown]
# ## What's the Same Between Reading a Sequence of Words and Reading a Story?
# Like CMR, models of comprehension are connectionist and conceptualize encoding new information as a dynamic, interactive set of processes including 
# 1) the activation of prior relevant knowledge, 
# 2) the integration of that activated information with representations of the current experience, and 
# 3) updates to both a drifting representation of recenct experience as well as associations encoded into a longer-term associative memory store. 

# %% [markdown]
# ## What's the Difference??
# **Convergence.** How strongly a piece of information gets activated depends on support received from related information. In memory, this drives already highly connected idea units in a story toward even greater connectivity by the end of reading, and vice versa for more isolated units. Dynamics like these don't emerge in a model like CMR.
#
# **Mapping.** Concrete linguistic structures in a story can relate idea units without in memory based on properties aside from temporal contiguity or even semantic similarity. Different models characterize this mapping process in different ways or with different degrees of sophistication, but all emphasize the importance of this process.
#
# **Coherence-Based Retrieval** Comprehension is considered roughly goal-directed, meaning-seeking, aiming at efficient integration of processed ideas into a coherent (logical/consistent) representation. Computationally, this has a couple consequences. First, when mapping between incoming and previously activated concepts during reading fail -- these are called "cohesion gaps" in the text -- readers tend to infer relations between text constituents automatically, inferring causal, spatial, and temporal relationships based on prior knowledge. This contributes to the convergence pattern noted above, as reactivation of relevant prior knowledge to understand the current text progressively enhances associations between 
#
# **Levels of comprehension**. Some models distinguish between surface, text-base, and situation model levels of text understanding. Differences in processes for resolving coherent representations of a text at each level, mediated by reader goals and abilities, are thought to differentially influence final associations between idea units during recall. Other models avoid enforcing these distinctions, but tend to either be very flexible (effectively leaving the implications of these differences to be weighed on a case by case basis through parameter configuration) or very simple. 

# %% [markdown]
# ## How Models Capture The Difference Between Stories and Word Sequences
# Substantive differences between models of text comprehension mainly revolve entirely around the kinds of linguistic features that drive mapping between **idea units.** For example, the construction-integration model emphasizes argument overlap between units, while the causal networking model emphasizes cause-effect relations within a story and models such as the structure-building and event-indexing model focused on how continuity or disruption thereof could sort stories into sequences of discrete events collecting groups of ideas. 
#
# The landscape model of reading comprehension is comparatively flexible/agnostic, and allows identification of relations between idea units to be contingent based on the premises of current research, and instead concerning itself with how pre-existing relations between concepts evolve over and a consequence of reading them together in a story.
