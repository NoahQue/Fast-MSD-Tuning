
# Fine-tuning on Multi-Source Datasets for Graph Neural Networks


## In-domain

### Datasets

Datasets for molecular property prediction can be found [here](https://github.com/snap-stanford/pretrain-gnns#dataset-download) (This [link](http://snap.stanford.edu/gnn-pretrain/data/chem_dataset.zip) for downloading).

### Requirements
```bash
pytorch >= 1.8.0
torch_geometric >= 2.0.3
rdkit >= 2019.03.1.0
tqdm >= 4.31
```

### fast finetuning
```bash
python fastfinetune_sequence.py
```


## Out-of-domain

### Requirements
```bash
Linux with Python ≥ 3.6
PyTorch ≥ 1.4.0
0.5 > DGL ≥ 0.4.3
conda install -c conda-forge rdkit=2019.09.2
`pip install -r requirements.txt`
```

### fast finetuning
```bash
python fastfinetune_sequence.py
```





