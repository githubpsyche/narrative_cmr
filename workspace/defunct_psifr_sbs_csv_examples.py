# %% [markdown]
# ## Examples

# %% [markdown]
# Then the data can easily be retrieved in one line, with `pivot_table` applied to retrieve trial arrays and whatnot. Right?

# %%
import pandas as pd

data = pd.read_csv('../data/psifr_sbs.csv')
data.head()

# %%
from psifr import fr
events = fr.merge_free_recall(data, list_keys=['item_index', 'cycle', 'story_index', 'story_name', 'time_test'])

events.head()

# %% [markdown]
# ### Trials Array

# %%
trials_df = events.pivot_table(index=['subject', 'list'], columns='output', values='input')
trials_df.head()

# %%
trials_df.to_numpy(na_value=0).astype('int64')

# %%
events.pivot_table(index=['subject', 'list'], columns='input', values='cycle')

# %% [markdown]
# And an example visualization:

# %%
from narrative_cmr.Temporal_Effects import visualize_aggregate
import matplotlib.pyplot as plt

for i in range(1, 4):
    visualize_aggregate(events.loc[(events.time_test==i)], data_query='subject > -1')
    plt.show()

# %%
for i in range(1, 4):
    print(fr.lag_rank(events[events.time_test==i]).agg(['mean', 'sem']))

# %%
import numpy as np

passage_names = ['Fisherman', 'Supermarket', 'Flight', 'Cat', 'Fog', 'Beach']

for i in range(1, 4):
    for story_name in passage_names:
        print(story_name, i)
        print(fr.distance_rank(events.loc[(events.story_name==story_name) & (events.time_test==i)], 'item_index', 
                           1-np.array(similarity_result[story_name])).agg(['mean', 'sem']))

# %%


edges = [0.5299, 0.5799, 0.6299, 0.6799, 0.7299, 0.7799, 0.8299, 0.8799, 0.9299, 0.9799]
centers = [0.5620, 0.6102, 0.6593, 0.7073, 0.7552, 0.8032, 0.8516, 0.9023, 0.9440]

for story_name in passage_names:
    
    dcrp = fr.distance_crp(events[events.story_name==story_name], 'item_index', 
                    1-np.array(similarity_result[story_name]), edges, centers)
    g = fr.plot_distance_crp(dcrp, min_samples=5)
    g.set(xlim=(.5, 1), ylim=(0, .2));

    plt.show()


# %%
