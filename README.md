# FCT: Fair Clasification Tree

A low-depth Classification Tree that is optimized for leaf accuracy.

After this, leaves of the tree are extended using further models. In this implementation, we extend using XGBoost models. However, we could choose any model in practie.

The hybrid tree combines the explainability of Classification Trees with the accuracy of XGBoost.

## Results
To generate visualizations and tables used in the paper see `Results.ipynb`.


## Requirements
Requirements are listed in the `requirements.txt` file. To install them, run ```pip install -r requirements.txt```

## Datasets
To download datasets, run ```python ./utils/openml_data_down.py```. This downloads the classification part of the tabular benchmark by Grinzstajn et al. to folders `./data/openml/categorical` and `./data/openml/numerical`.

## Usage
Regarding the configurations that were executed, they are listed in `benchmark.py`
That script cannot be run, it is made for use on a cluster where the experiments were executed.


### Hybrid-trees
A simple proof-of-concept example is in `Example.ipynb` with a walkthrough.

To run the proper optimization yourself, follow these 2 steps:
 - Run `python sklearn_warmstart.py -data path/to/data -res path/to/results` This will compute the low-depth tree for the data with default parameters, as presented in the paper. Data must be in the same format that is used in the download script. Results folder must be created in advance. For different hyperparameters, refer to the python implementation. This creates a `run0.ctx` in the `path/to/results` folder. This is the context representing the model (0 in the name is the seed used). You can choose different strategy of optimization by selecting different python script from `sklearn_warmstart.py`. The options are:
    - `sklearn_warmstart.py` for Warmstrarted variant
    - `gradual.py` for Gradual variant
    - `direct.py` for Direct variant
    - `halving.py` for Halving variant - not used in the paper, an earlier version using bisection, described in the thesis
    - `oct.py` for OCT
 - Then, run `python finalize_model.py path/to/results/[model].ctx` to extend the tree stored in `[model].ctx` file with XGBoost models in leaves. The hybrid tree will be saved in a `[model]_ext.ctx`. This file can then be loaded using the functions in `src/UtilityHelper.py`

To investigate how were the results collected, see the function `retrieve_information()` in `src/UtilityHelper.py`

### CART
To run the optimization of CART methods, run the following: `python find_best_trees.py` This generates a file per configuration, containing the same information as used in the `Results.ipynb`

Note that the script takes a lot of time to finish, and can take a lot of memory as well.

## Reference

Improving the Validity of Decision Trees as Explanations \
*__Jiří Němeček__, Tomáš Pevný, Jakub Mareček* \
[link to the preprint](https://arxiv.org/abs/2306.06777)


## Master's Thesis
This repository is a major part of Jiří Němeček's [Master's Thesis](https://dspace.cvut.cz/handle/10467/109455?locale-attribute=en) at FEE CTU in Prague
