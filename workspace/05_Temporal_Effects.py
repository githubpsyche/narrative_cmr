# %% [markdown]
# # Temporal Effects

# %% [markdown]
# We lean on the `psifr` toolbox to generate three plots corresponding to the contents of Figure 4 in Morton & Polyn, 2016:
# 1. Recall probability as a function of serial position
# 2. Probability of starting recall with each serial position
# 3. Conditional response probability as a function of lag
#
# Input data is presumed to be [formatted for use of the psifr toolbox](https://psifr.readthedocs.io/en/latest/guide/import.html). For demos, we'll import `psifr_sbs.csv`.

# %%
import pandas as pd
from psifr import fr

data = pd.read_csv('data/psifr_sbs.csv')
events = fr.merge_free_recall(data, list_keys=['item_index', 'cycle', 'story_index',
                                               'story_name', 'time_test'])

events.head()

# %% [markdown]
# ##  Serial Position Curves

# %% [markdown]
# `psifr` includes a lot of functions for visualization, but if your cookbook depends on them you'll get considerably less flexibility over the appearance and other features of your plots. I prefer to use the library to generate applicable DataFrames and then plot them myself.

# %%
from psifr import fr

fr.spc(events)

# %%
fr.plot_spc(fr.spc(events).reset_index())

# %% [markdown]
# What is the structure of this DataFrame? For each subject, the DataFrame tracks the rate at which items with each input position were recalled across all recorded trials. We can visualize these values aggregated across subject like so:

# %%
import seaborn as sns

g = sns.FacetGrid(fr.spc(events))
g.map_dataframe(sns.lineplot, 'input', 'recall');

# %% [markdown]
# ### Story_Name Facets

# %% [markdown]
# But there are different numbers of idea units in different trials. The DataFrame retrieved when I call `fr.spc` doesn't include the `story_name` column indexing this factor, so I probably have to move beyond the provided library. Below we first pivot `events` to create a DataFrame like the output of `fr.spc`, but sorted by story name. Then we convert the tracked story names into a column of this DataFrame so that plots for each condition can be laid out on a FacetGrid.

# %%
clean = events.query('study').pivot_table(index=['story_name', 'input'], values=['recall'])
clean.reset_index(level=0, inplace=True)
clean

# %%
g = sns.FacetGrid(clean, col='story_name')
g.map_dataframe(sns.lineplot, 'input', 'recall');

# %% [markdown]
# ### Subject-Level Curves

# %% [markdown]
# What if we wanted to visualize each subject's SPC for each story_name? To do this, we just have to aggregate over and then convert into a column another variable:

# %%
clean = events.query('study').pivot_table(index=['subject', 'story_name', 'input'], values=['recall'])
clean.reset_index(level=0, inplace=True)
clean.reset_index(level=0, inplace=True)

g = sns.FacetGrid(clean, col='story_name')
g.map_dataframe(sns.lineplot, 'input', 'recall', hue='subject');

# %% [markdown]
# And then we could view aggregated SPCs by each subject facet instead of story_name:

# %%
clean = events.query('study').pivot_table(index=['subject', 'input'], values=['recall'])
clean.reset_index(level=0, inplace=True)

g = sns.FacetGrid(clean, col='subject')
g.map_dataframe(sns.lineplot, 'input', 'recall');

# %% [markdown]
# ### Automatic Confidence Intervals

# %% [markdown]
# Seaborn's `lineplot` function also automatically computes confidence intervals over any additional factors you include. For example, it might be possible to include a `list` column in our calculations if we want to see something like a 95% confidence interval over each trial in our facet. The original `fr.spc` shows a confidence interval over subject, so we do that here:

# %%
clean = events.query('study').pivot_table(index=['subject', 'story_name', 'input'], values=['recall'])
clean.reset_index(inplace=True)

g = sns.FacetGrid(clean, col='story_name')
g.map_dataframe(sns.lineplot, 'input', 'recall');

# %% [markdown]
# What have we learned? Dataframes useful for visualizing a serial position curve are easy enough to generate that resorting to `psifr` to do it is often unnecessary. When you'll want to generate serial position curves over different conditions in your dataset, it's often worth it. Applying `pivot_table` with your condition as one of the indices and then applying `reset_index` to convert your indices into columns creates a DataFrame you can easily generate these visualizations over.

# %% [markdown]
# ## Lag CRPs
# Because serial position curves amount to a mean over values in one column split between each condition in a different column, generating them is well-supported by pandas and seaborn. But what about more domain-specific analyses like our lag CRP?
#
# Again, `psifr` includes a function tracking these statistics out of the box:

# %%
fr.lag_crp(events)

# %%
max_lag = 5
filt_neg = f'{-max_lag} <= lag < 0'
filt_pos = f'0 < lag <= {max_lag}'

g = sns.FacetGrid(fr.lag_crp(events).reset_index())
g.map_dataframe(
    lambda data, **kws: sns.lineplot(
        data=data.query(filt_neg), x='lag', y='prob', **kws)
)
g.map_dataframe(
    lambda data, **kws: sns.lineplot(
        data=data.query(filt_pos), x='lag', y='prob', **kws)
)
g.set_xlabels('Lag')
g.set_ylabels('CRP')
g.set(ylim=(0, 1));

# %% [markdown]
# A 95% confidence interval is plotted over each subject just like with our basic SPC.

# %% [markdown]
# ### Story_Name Facets

# %% [markdown]
# We similarly want to be able to see a corresponding plot for each story_name. But it obviously will take more work to get there. Instead of trying to replace `psifr.lag_crp` with an in-house implementation, we'll just apply it for different parts of our dataset and then aggregate the resulting DataFrame. This is also an option for using psifr to generate SPCs if we're not interested in writing our own code that tracks each factor:

# %%
lag_crps = []
for story_name in pd.unique(events.story_name):
    lag_crps.append(fr.lag_crp(events[events.story_name == story_name]))
    
clean = pd.concat(lag_crps, keys=pd.unique(events.story_name), names=['story_name']).reset_index()
clean

# %%
max_lag = 5
filt_neg = f'{-max_lag} <= lag < 0'
filt_pos = f'0 < lag <= {max_lag}'

g = sns.FacetGrid(clean, col='story_name')
g.map_dataframe(
    lambda data, **kws: sns.lineplot(
        data=data.query(filt_neg), x='lag', y='prob', **kws)
)
g.map_dataframe(
    lambda data, **kws: sns.lineplot(
        data=data.query(filt_pos), x='lag', y='prob', **kws)
)
g.set_xlabels('Lag')
g.set_ylabels('CRP')
g.set(ylim=(0, 1));

# %%
max_lag = 5
filt_neg = f'{-max_lag} <= lag < 0'
filt_pos = f'0 < lag <= {max_lag}'

g = sns.FacetGrid(clean)
g.map_dataframe(
    lambda data, **kws: sns.lineplot(
        data=data.query(filt_neg), x='lag', y='prob', hue='story_name', **kws)
)
g.map_dataframe(
    lambda data, **kws: sns.lineplot(
        data=data.query(filt_pos), x='lag', y='prob', hue='story_name', **kws)
)
g.set_xlabels('Lag')
g.set_ylabels('CRP')
g.set(ylim=(0, 1));

# %% [markdown]
# This approach gives us the full dataframe we're interested in, giving us room to customize plotting outputs ourselves. 

# %% [markdown]
# ## Lag Rank
# As explained in the psifr documentation:
#
# > We can summarize the tendency to group together nearby items using a lag rank analysis. For each recall, this determines the absolute lag of all remaining items available for recall and then calculates their percentile rank. Then the rank of the actual transition made is taken, scaled to vary between 0 (furthest item chosen) and 1 (nearest item chosen). Chance clustering will be 0.5; clustering above that value is evidence of a temporal contiguity effect.

# %%
fr.lag_rank(events)

# %% [markdown]
# ### T-Testing
# Papers using this metric have applied one-sample t-tests over each subject to test the null hypothesis that no clustering is happening on the basis of chance (mean = .5). Since .5 is the floor output for this analysis, we're doing a one-tailed analysis.

# %%
from scipy import stats

stats.ttest_1samp(fr.lag_rank(events), .5, alternative='greater')

# %% [markdown]
# ### Story_Name Facets

# %% [markdown]
# An approach similar to our handling of CRPs across facets is okay:

# %%
lag_ranks = []
for story_name in pd.unique(events.story_name):
    lag_ranks.append(fr.lag_rank(events[events.story_name == story_name]))
    
clean = pd.concat(lag_ranks, keys=pd.unique(events.story_name), names=['story_name']).reset_index()
clean

# %%
import matplotlib.pyplot as plt

g = sns.catplot(x='story_name', y='rank', data=clean, kind='violin');
sns.swarmplot(x='story_name', y='rank', data=clean, ax=g.ax, color="k", size=3);

g.set_xticklabels(rotation=20, ha="right")
plt.tight_layout()
plt.xlabel('')
plt.ylabel('lag-rank')
plt.show()

# %% [markdown]
# The exact plot I want depends so much on the context that I'll just use the psifr library to help generate DataFrames and transform from there.
