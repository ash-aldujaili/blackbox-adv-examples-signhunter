
### Setup


#### A. Environment Setup
```
conda env create -f ./requirements/conda_env.yaml
```

Before running any of the scripts. Make sure the environment is activated.
```
conda activate sign4bb-venv
```

To deactivate:
```
deactivate
```

For the rest of this document, it is assumed that `pwd` is `src/`

#### B.Pre-trained models & Datasets Setup

```
export PYTHONPATH=`pwd`
python utils/download_models.py # downloads mnist and cifar10 models
python utils/download_cifar10.py # download cifar10 dataset
```
MNIST will be downloaded upon first run of the code. 
For IMAGENET, please download the imagenet validation set and make sure the foloder is structured in the following:

### Experiments

#### A. To run black-box attack experiments:

After modifying the attacks config files in `config-jsons` as desired; include config files of the considered attacks in `attacks/blackbox/run_attack.py`'s main function. Make sure that the `IS_DEBUG_MODE` is set to False. This Flag is used to replace the repsective model's loss with the synthetic concave one discussed in Appendix D in the supplement.

```
python run_attack.py
```

#### B. To run Keep K experiment

```
python ./experiments/keep_k_sign_exp.py
```

#### C. To run SignHunter for Continuous Optimization Experiments

`cd` to `src/experiments`

```
python signhunter_4_cont_opt.py
```

#### D. Rest of the Paper/Supplement Experiments

`cd` to `src/notebooks`

```
jupyter notebook
```

And go through the cells of `approx-hamming-search.ipynb`


### Other Utilities

To generate algorithms' hyperparameter tables (Appendix D):

```
# to export parameter table of algorithms
json_files = get_jsonfiles('*bandit*')
utils.latex_fcts.config_json_2_tbl_latex(json_files)
```

TeX files will be generated in `data/tex_tbls`

---

To view experiments based on Tensorboard:
Currently, `hamming_dist_exp`, `partial_derivatives_exp` export their results to Tensoboard

```
tensorboard --logdir=../data/hamming_dist_exp/
```

---

To generate plots/tables from `attacks/blackbox/run_attack.py`

`run_attack.py` will store results into both `h5` file (logging progress of the algorithms) as well as a json (logging the final status of the algorithms)

To generate plots (e.g., Figures in Appdendix E) from data stored in the resulting `h5` file

```
utils.plt_fcts.plt_from_h5tbl(['../../data/blackbox_attack_exp/{H5_FILE_NAME}.h5'])
```

Plots will be generated in `data/blackbox_attack_exp/{H5_FILE_NAME}_plots`

---

To generate summary of attack latex tables (e.g., Tables in Appdendix E) from the resulting `json` file
```
utils.latex_fcts.res_json_2_tbl_latex('../../data/blackbox_attack_exp/{JSON_FILE_NAME}.json')
```

TeX files will be generated in `data/blacbox_attack_exp`

---

To run the experiments of Section 6, checkout to the `challenges` branch

```
git checkout challenges
git pull
```

I've put these on a branch since the same code and config files used in main experiments are modified to generate files in the format required by the challenges.






