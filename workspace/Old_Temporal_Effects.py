# %% [markdown]
# # Temporal Effects

# %% [markdown]
# We lean on the `psifr` toolbox to generate three plots corresponding to the contents of Figure 4 in Morton & Polyn, 2016:
# 1. Recall probability as a function of serial position
# 2. Probability of starting recall with each serial position
# 3. Conditional response probability as a function of lag
#
# Input data is presumed to be [formatted for use of the psifr toolbox](https://psifr.readthedocs.io/en/latest/guide/import.html). 

# %%
# export

import pandas as pd
import seaborn as sns
from psifr import fr
import matplotlib.pyplot as plt

def visualize_individuals(data, data_query='subject > -1'):
    """
    Visualize variation between subjects in dataset wrt key organizational metrics.
    """

    # generate data-based spc, pnr, lag_crp
    data_spc = fr.spc(data).query(data_query).reset_index()
    data_pfr = fr.pnr(data).query('output <= 1').query(data_query).reset_index()
    data_lag_crp = fr.lag_crp(data).query(data_query).reset_index()

    # spc
    g = sns.FacetGrid(dropna=False, data=data_spc)
    g.map_dataframe(sns.lineplot, x='input', y='recall', hue='subject')
    g.set_xlabels('Serial position')
    g.set_ylabels('Recall probability')
    #plt.title('Recall Probability by Serial Position')
    g.set(ylim=(0, 1))
    plt.savefig('spc.pdf', bbox_inches='tight')

    # pfr
    h = sns.FacetGrid(dropna=False, data=data_pfr)
    h.map_dataframe(sns.lineplot, x='input', y='prob', hue='subject')
    h.set_xlabels('Serial position')
    h.set_ylabels('Probability of First Recall')
    #plt.title('P(First Recall) by Serial Position')
    h.set(ylim=(0, 1))
    plt.savefig('pfr.pdf', bbox_inches='tight')

    # lag crp
    max_lag = 5
    filt_neg = f'{-max_lag} <= lag < 0'
    filt_pos = f'0 < lag <= {max_lag}'
    i = sns.FacetGrid(dropna=False, data=data_lag_crp)
    i.map_dataframe(
        lambda data, **kws: sns.lineplot(
            data=data.query(filt_neg), x='lag', y='prob', hue='subject', **kws))
    i.map_dataframe(
        lambda data, **kws: sns.lineplot(
            data=data.query(filt_pos), x='lag', y='prob', hue='subject', **kws))
    i.set_xlabels('Item Lag')
    i.set_ylabels('Conditional Response Probability')
    #plt.title('Recall Probability by Item Lag')
    i.set(ylim=(0, 1))
    plt.savefig('crp.pdf', bbox_inches='tight')


# %%
#export

import pandas as pd
import seaborn as sns
from psifr import fr
import matplotlib.pyplot as plt

def visualize_aggregate(data, data_query):
    
    # generate data-based spc, pnr, lag_crp
    data_spc = fr.spc(data).query(data_query).reset_index()
    data_pfr = fr.pnr(data).query(
        'output <= 1').query(data_query).reset_index()
    data_lag_crp = fr.lag_crp(data).query(data_query).reset_index()

    # spc
    g = sns.FacetGrid(dropna=False, data=data_spc)
    g.map_dataframe(sns.lineplot, x='input', y='recall',)
    g.set_xlabels('Serial position')
    g.set_ylabels('Recall probability')
    #plt.title('Recall Probability by Serial Position')
    g.set(ylim=(0, 1))
    plt.savefig('spc.pdf', bbox_inches='tight')

    # pfr
    h = sns.FacetGrid(dropna=False, data=data_pfr)
    h.map_dataframe(sns.lineplot, x='input', y='prob')
    h.set_xlabels('Serial position')
    h.set_ylabels('Probability of First Recall')
    #plt.title('P(First Recall) by Serial Position')
    h.set(ylim=(0, 1))
    plt.savefig('pfr.pdf', bbox_inches='tight')

    # lag crp
    max_lag = 5
    filt_neg = f'{-max_lag} <= lag < 0'
    filt_pos = f'0 < lag <= {max_lag}'
    i = sns.FacetGrid(dropna=False, data=data_lag_crp)
    i.map_dataframe(
        lambda data, **kws: sns.lineplot(data=data.query(filt_neg),
                                         x='lag', y='prob', **kws))
    i.map_dataframe(
        lambda data, **kws: sns.lineplot(data=data.query(filt_pos),
                                         x='lag', y='prob', **kws))
    i.set_xlabels('Item Lag')
    i.set_ylabels('Conditional Response Probability')
    #plt.title('Recall Probability by Item Lag')
    i.set(ylim=(0, 1))
    plt.savefig('crp.pdf', bbox_inches='tight')


# %%
# export

def visualize_model(
    model, simulation_function, experiment_count, first_recall_item=None):
    
    """
    Visualize aggregate using simulated data associated with this model.
    """
    
    visualize_aggregate(simulation_function(
        model, experiment_count, first_recall_item), data_query)


# %%
# export

import pandas as pd
import seaborn as sns
from psifr import fr
import matplotlib.pyplot as plt

def visualize_fit(
    model_class, parameters, data, data_query=None, experiment_count=1000, savefig=False):
    
    """
    Apply organizational analyses to visually compare the behavior of the model 
    with these parameters against specified dataset.
    """
    
    # generate simulation data from model
    model = model_class(**parameters)
    sim_data = simulate_data(model, experiment_count)
    
    # generate simulation-based spc, pnr, lag_crp
    sim_spc = fr.spc(sim_data).reset_index()
    sim_pfr = fr.pnr(sim_data).query('output <= 1') .reset_index()
    sim_lag_crp = fr.lag_crp(sim_data).reset_index()
    
    # generate data-based spc, pnr, lag_crp
    data_spc = fr.spc(data).query(data_query).reset_index()
    data_pfr = fr.pnr(data).query('output <= 1').query(data_query).reset_index()
    data_lag_crp = fr.lag_crp(data).query(data_query).reset_index()
    
    # combine representations
    data_spc['Source'] = 'Data'
    sim_spc['Source'] = model_class.__name__
    combined_spc = pd.concat([data_spc, sim_spc], axis=0)
    
    data_pfr['Source'] = 'Data'
    sim_pfr['Source'] = model_class.__name__
    combined_pfr = pd.concat([data_pfr, sim_pfr], axis=0)
    
    data_lag_crp['Source'] = 'Data'
    sim_lag_crp['Source'] = model_class.__name__
    combined_lag_crp = pd.concat([data_lag_crp, sim_lag_crp], axis=0)
    
    # generate plots of result
    # spc
    g = sns.FacetGrid(dropna=False, data=combined_spc)
    g.map_dataframe(sns.lineplot, x='input', y='recall', hue='Source')
    g.set_xlabels('Serial position')
    g.set_ylabels('Recall probability')
    #plt.title('Recall Probability by Serial Position')
    g.add_legend()
    g.set(ylim=(0, 1))
    plt.savefig('{}_fit_spc.pdf'.format(model_class.__name__), bbox_inches='tight')
    
    #pdf        
    h = sns.FacetGrid(dropna=False, data=data_pfr)
    h.map_dataframe(sns.lineplot, x='input', y='prob', hue='Source')
    h.set_xlabels('Serial position')
    h.set_ylabels('Probability of First Recall')
    #plt.title('P(First Recall) by Serial Position')
    h.add_legend()
    h.set(ylim=(0, 1))
    plt.savefig('{}_fit_pfr.pdf'.format(model_class.__name__), bbox_inches='tight')
    
    # lag crp        
    max_lag = 5
    filt_neg = f'{-max_lag} <= lag < 0'
    filt_pos = f'0 < lag <= {max_lag}'
    i = sns.FacetGrid(dropna=False, data=data_lag_crp)
    i.map_dataframe(
        lambda data, **kws: sns.lineplot(data=data.query(filt_neg),
                                         x='lag', y='prob', hue='Source', **kws))
    i.map_dataframe(
        lambda data, **kws: sns.lineplot(data=data.query(filt_pos),
                                         x='lag', y='prob', hue='Source', **kws))
    i.set_xlabels('Item Lag')
    i.set_ylabels('Conditional Response Probability')
    #plt.title('Recall Probability by Item Lag')
    i.add_legend()
    i.set(ylim=(0, 1))
    if savefig:
        plt.savefig('{}_fit_crp.pdf'.format(model_class.__name__), bbox_inches='tight')
    else:
        plt.show()

# %%
