# DiffAffinity

![text](./figures/RDGM.png)

Predicting mutational effects on protein-protein
binding via a side-chain diffusion probabilistic model.
Shiwei Liu*, Tian Zhu*, Milong Ren, Dongbo Bu, Haicang Zhang#
*These authors contribute equally.
#correspondence should be addressed to H.Z.

[Paper link on NeurIPS 2023](https://neurips.cc/virtual/2023/poster/72495)


## Install

### DiffAffinity Environment

```bash
git clone https://github.com/oxcsml/geomstats.git 
conda create -n DiffAffinity python=3.9
conda activate DiffAffinity
pip install -r requirements.txt
pip install jaxlib==0.4.1+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
GEOMSTATS_BACKEND=jax pip install -e geomstats
pip install -e .
```

- `requirements.txt` contains the core requirements for running the code in the `riemmanian_score_sde` and `DiffAffinity` packages. NOTE: you may need to alter the jax versions here to match your setup.



## Datasets
Protein Structures and SKEMPI v2 dataset can be downloaded use following scripts.  
| Dataset   | Download Script                                    |
| --------- | -------------------------------------------------- |
| [PDB-REDO](https://pdb-redo.eu/)  | [`data/get_pdbredo.sh`](./data/get_pdbredo.sh)     |
| [SKEMPI v2](https://life.bsc.es/pid/skempi2) | [`data/get_skempi_v2.sh`](./data/get_skempi_v2.sh) |

The `data` folder contains the data for downstream experiments. 
 | File   | Useage                                |
| --------- | -------------------------------------------------- |
| [DDG_6m0j.csv](./data/DDG_6m0j.csv)  | Free energy changes ($\Delta \Delta G$) of all 285 possible single-point mutations on Wuhan-Hu-1 RBD (PDB ID: 6M0J)|
| [6m0j.pdb](./data/6m0j.pdb) | Protein experimental structure of Wuhan-Hu-1 RBD (pid: 6M0J)|
| [7FAE_RBD_Fv.pdb](./data/7FAE_RBD_Fv.pdb) | Protein experimental structure of  the SARS-CoV-2 virus spike protein (pid: 7FAE)|

## Trained Weights
Trained model weights are available here [Google Driver](https://drive.google.com/drive/folders/1NmKl-mLVgwBP7IVwX6BkJ2mB5xIYjRKB?usp=sharing) 

## Usage

### Train Side-chain Diffusion Probabilistic Model SidechainDiff

```bash
bash SidechainDiff_train.sh
```
###  Prediction of Side-chain Conformations

```bash
bash SidechainDiff_test.sh
```
Sample results of torsion angles can be found in `workdir`. 

[parse_atom14.ipynb](./parse_atom14.ipynb) can transform torsion angles to atom14 coordinates. 

---------
### Train Mutational Effect Predictor DiffAffinity 

```bash
python DiffAffinity.py ./context_generator/configs/train/da_ddg_skempi.yml --idx_cvfolds 0
```
The `--idx_cvfolds` is optional (0,1,2) and the default setting is 0

Because we have three DiffAffinity models, you need to change the checkpoint in [6m0j.yml](./context_generator/configs/inference/6m0j.yml) and [7FAE_RBD_Fv_mutation.yml](./context_generator/configs/inference/7FAE_RBD_Fv_mutation.yml) for downstream tasks

### Predicting Mutational Effects On Binding Affinity of SARS-CoV-2 RBD
```bash 
python prediction.py ./context_generator/configs/inference/6m0j.yml
```

### Predict Mutational Effects for a SARS-CoV-2 Human Antibody and Other Protein Complexes

```bash
python prediction.py ./context_generator/configs/inference/7FAE_RBD_Fv_mutation.yml
```

# Acknowledgements
We acknowledge that the part of the Riemainn SDE code is adapted from [riemannian-score-sde](https://github.com/oxcsml/riemannian-score-sde/tree/main). Thanks to the authors for sharing their code. 





