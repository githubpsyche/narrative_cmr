# %% [markdown]
# # Project Tools
# Helper functions to help manage research projects. I might ditch them, who knows?

# %% [markdown]
# ## Pandoc to Cell Output Configuration

# %% [markdown]
# I want a function that selects a latex file at bibliography and raters its content as formatted markdown within a Jupyter notebook.

# %%
from IPython.display import Markdown

def render_tex(tex_path, bib_path, csl_path):
    result = !pandoc -C --ascii {tex_path} -f latex -t markdown_mmd --bibliography {bib_path} --csl {csl_path}
    return Markdown('\n'.join(result))


# %% [markdown]
# To test the function, we'll render the paper abstract inside `writing/`. 

# %%
render_tex('writing/01_Abstract.tex', 'writing/references.bib', 'writing/main/apa.csl')

# %% [markdown]
# ## Export Parts of Notebooks to Importable Python Scripts

# %%
from nbdevminimum.core import simple_export_all_nb
from pathlib import Path

simple_export_all_nb(nbs_path=Path('.'), lib_path=Path('narrative_cmr'))

# %%
