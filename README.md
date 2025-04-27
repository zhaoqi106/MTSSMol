## A multi-task self-supervised strategy for predicting molecular properties and FGFR1 inhibitors ##



## Getting Started

### Installation

Set up conda environment and clone the github project

```
# create a new environment
$ conda create --name MTSSMol python=3.7
$ conda activate MTSSMol

# install requirements
$ pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 -f https://download.pytorch.org/whl/torch_stable.html
$ pip install torch-geometric==1.6.3 torch-sparse==0.6.9 torch-scatter==2.0.6 -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
$ pip install PyYAML
$ conda install -c conda-forge rdkit=2020.09.1.0
$ conda install -c conda-forge tensorboard
$ conda install -c conda-forge nvidia-apex # optional

# clone the source code of MTSSMol
$ git clone https:// github.com/zhaoqi106/MTSSMol
$ cd MTSSMol
```

### Dataset

Data for pre-training can be obtained by contacting the author. All the databases for fine-tuning are saved in the folder under the benchmark name. You can also find the benchmarks from [MoleculeNet](https://moleculenet.org/).

### Prepare dataset
Assign pseudo-labels to the data：
```
$ python pseudo_label_process.py.py
```

### Pre-training

To train the MTSSMol, where the configurations and detailed explaination for each variable can be found in `config.yaml`
```
$ python MTSSMol.py
```

To monitor the training via tensorboard, run `tensorboard --logdir ckpt/{PATH}` and click the URL http://127.0.0.1:6006/.

### Fine-tuning 

To fine-tune the MTSSMol pre-trained model on downstream molecular benchmarks, where the configurations and detailed explaination for each variable can be found in `config_finetune.yaml`
```
$ python finetune.py
```
### Evaluation
We provided a prediction method for the FGFR1 dataset，then the results can be reproduced by:
```
$ python evaluation.py
```
### Pre-trained models

We also provide pre-trained  GIN models, which can be found in `ckpt/pretrained_gin`. 
