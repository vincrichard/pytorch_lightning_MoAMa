
## Repository information

This is a reimplementation of the paper [Motif-aware Attribute Masking for Molecular Graph Pre-training](https://arxiv.org/abs/2309.04589)

You can find the original code of the authors [here](https://github.com/einae-nd/MoAMa-dev)


The goal of this repository is to try to reproduce the result and having a denser implementation.
It contains the following:

- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/) based implementation of the pretraining and finetuning.

    Note that for the finetuning only the Tox21 dataset is implemented, but other dataset can easily be added.

- Dataset: Zinc and Tox21 are present in datasets.zip. It also contains the scaffold splits for the Tox21 dataset.

- Pretrained models: You can find the original authors models in their repository. We also provided my own trained model.

- Implementation of another motif generation. This is outside the range of the paper. We found the motif implementation limited and wanted to try another implementation.


The original code was not usable due to unoptimized code. With my hardware, it took
over a day per epoch. We tried to improve the speed of the code with the following modifications:

- Compute Motifs and Fingerprint before training. Those operations are time-consuming so here,
    we compute it before training and save them for future use.
- Compute the inter batch similarity in a custom collator function. Again time-consuming so making use
    of the multiple worker of the PyTorch Dataloader to gain some time.
- Vectorize operation, where possible We tried to vectorize the unoptimized code.

Overall those change add a x10 time boost on my hardware. However, We were not able
to reproduce the result of the original shared weights. So there might be a bug somewhere.

This project went down on my priority and even with the authors code We were not able to attain
the paper results (with the authors' inference code see [here](https://github.com/einae-nd/MoAMa-dev/issues/4)). So we don't plan on pursuing this project anymore.

All experiment runs logs and models (except finetuning weights) can be found in the experiment_logs.zip archive. Datasets are in the data/dataset.zip archive.

### Results

After 1 pretraining on Zinc, and 10 runs on the finetuning of the Tox21 dataset.

- Weights shared by the authors: `0.743 ± 0.007`
- Retrain MoAMa model with this repository: `0.727 ± 0.007`
- MoAMa with custom extractor: `0.730 ± 0.005`

Note if we ever retake this project: these are the results on this code inference. There seems to be a bug here
as well, we get better result when predicting with the code of the authors.
We are not sure why, the pipeline is quite similar to GraphMAE and we reused most of the code
from it. We have another repo with GraphMAE code which predicts the expected results.


## Installation

To install a working environment you can run the following.

```bash
conda env create -f environment.yaml -p ./.env
```

## Run the pretraining

You can run the following:

```bash
python -m src.pretrain data/config/pretraining_moama.py
```

It will run the pretraining and save the logs in `logs/MoAMa/pretrain`.


## Run the finetuning

```bash
python -m src.finetuning data/config/finetuning_tox21.py
```

It will train 10 models with the encoder of the pretrained model. You can find a mean / avg of the result at the end of the `finetuning.log` file.


## Note on the configuration

In this repository we are experimenting with a custom configuration setup. It mostly follows
the [pure python mmengine configuration](https://mmengine.readthedocs.io/en/latest/advanced_tutorials/config.html#a-pure-python-style-configuration-file-beta)
where we adapted it to be able to build the instance with a `build` method.
But the mmengine documentation should let you understand the overall functionality.

## Reason for Custom extractor

We found the original extractor to be limited compared to the original paper explanation (see this [issue](https://github.com/einae-nd/MoAMa-dev/issues/3)),
We tried a different implementation which tries to be closer to the original claim.
In this case the `BricsRingMotifExtractor` extracts all BRICS motif as well as all
separate rings. All single atom motif were removed, and we made sure that additional
random atom selected does not create motifs larger than the number of layer in the model.
