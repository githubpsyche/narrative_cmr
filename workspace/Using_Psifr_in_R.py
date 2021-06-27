# %% [markdown]
# # Using Psifr in R
# If you're doing memory research, like making your plots in R, but don't feel like coding up all the analyses yourself in that language, one option is use R's `reticulate` package to bring in a Python library to do the job. The `psifr` library for analysis and visualization of free recall data is one of the libraries you might be interested in using within R. 
#
# Even though it is written in Python, `psifr` is well-suited for R because its numerical outputs are Data Frames, rather than vectors or arrays as MATLAB-based functions tend to output. When using `reticulate`, R data frames are automatically converted to and from Python DataFrames. Moreover, the data format required for performing psifr analyses is well suited for the kinds of operations you'll be doing with `ggplot` to make visualizations. These features make use of the library relatively seamless.

# %% [markdown]
# ## Setting Up Reticulate and Psifr
# A full guide for this step can be found [here](https://rstudio.github.io/reticulate/)
#
# To make `reticulate` work, you need to already have an installation of Python somewhere on your computer. R might be able to find your installation automatically, but if not you might have to know where your Python install is located. You should also already have `psifr` installed as a Python library. 
#
# To do this, use pip in your terminal:
#
# ```python
# pip install psifr
# ```

# %% [markdown]
# ### Installing Reticulate
# In R, install `reticulate` like so:

# %%
install.packages("reticulate")

# %% [markdown]
# ### Starting Reticulate With a Specified Python Version

# %%
library(reticulate)
use_python("C:/ProgramData/Miniconda3/python")

# %% [markdown]
# ## Using Psifr (or any other python package) in R
# You can use the import() function to import any Python module and call it from R. `psifr` includes various submodules (with `fr` being the one most typically relevant for our research), so we technically import `psifr.fr` here.

# %% [markdown]
# ### Importing the Library and Some Data

# %%
fr <- import("psifr.fr")

# %% [markdown]
# To demo the library, we'll need some data. I'll use a csv generated from the results of Cutler, Palan, Brown-Schmidt, & Polyn (2019). 

# %%
data = read.csv("C:\\Users\\gunnj\\narrative_cmr\\data\\psifr_sbs.csv")
head(data)

# %% [markdown]
# More information about formatting data frames for compatibility with psifr can be found [here](https://psifr.readthedocs.io/en/latest/guide/import.html). I'd really recommend formatting your data this way in general, whether you'll be using `psifr` extensively or not, because data in this format can be transformed pretty easily to carry out most analyses. Either way, committing to a consistent format makes it easier to re-use your code across projects/datasets.

# %% [markdown]
# ### Serial Position Curve

# %% [markdown]
# With `psifr`, generating a DataFrame useful for plotting a serial position curve of your data can be done in two lines of code: 

# %%
events <- fr$merge_free_recall(data, list_keys=list('item_index', 'cycle', 'story_index',
                                               'story_name', 'time_test'))
head(fr$spc(events))

# %%
head(events)

# %%
