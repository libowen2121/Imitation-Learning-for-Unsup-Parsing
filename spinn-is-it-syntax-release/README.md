# Code for Williams, Drozdov, and Bowman '17

This release contains the source code for the paper: "[Do latent tree learning models identify meaningful structure in sentences?][11]"

It is built on a fairly large and unwieldy [codebase][9] that was prepared for the paper [A Fast Unified Model for Sentence Parsing and Understanding][1]. The master branch may still be under active development for other projects.

### Installation

Requirements:

- Python 2.7
- Pytorch 0.2.0

Install most required Python dependencies using the command below.

    pip install -r python/requirements.txt

Install Pytorch based on instructions online: `http://pytorch.org`

Data to download:

- 840B word 300D [GloVe word vectors](http://nlp.stanford.edu/projects/glove/)
- [MultiNLI](http://www.nyu.edu/projects/bowman/multinli/)
- [SNLI](http://nlp.stanford.edu/projects/snli/)

### Running new experiments

The main executables for the SNLI experiments in the paper is [`supervised_classifier.py`](https://github.com/nyu-mll/spinn/blob/is-it-syntax-release/python/spinn/models/supervised_classifier.py) (and the analogous [`rl_classifier.py`](https://github.com/nyu-mll/spinn/blob/is-it-syntax-release/python/spinn/models/rl_classifier.py), whose flags specify the hyperparameters of the model. You can specify gpu usage by setting `--gpu` flag greater than or equal to 0. Uses the CPU by default.

Here's a sample command that runs a fast, low-dimensional CPU training run, training and testing only on the dev set. It assumes that you have a copy of [SNLI](http://nlp.stanford.edu/projects/snli/) available locally.

    PYTHONPATH=spinn/python \
        python2.7 -m spinn.models.supervised_classifier --data_type nli \
        --training_data_path ~/data/snli_1.0/snli_1.0_dev.jsonl \
        --eval_data_path ~/data/snli_1.0/snli_1.0_dev.jsonl \
        --embedding_data_path python/spinn/tests/test_embedding_matrix.5d.txt \
        --word_embedding_dim 5 --model_dim 10 --model_type CBOW

### Retraining the experiments in the paper

Concatenate the SNLI and MultiNLI training sets into a single file, place it in an accessible directory, then run the commands in [`scripts/train_all_models_x5.sh`](https://github.com/nyu-mll/spinn/blob/is-it-syntax-release/scripts/train_all_models_x5.sh).

The experiment names don't neatly correspond to the terms used in the paper, but they shouldn't be too opaque. For a key: `noenc` = w/o Leaf LSTM, `enc_fix` = w/ Leaf LSTM, `_t_` = 'NC', `ChoiPyramid(_b)` = ST-Gumbel, `_s_` = Trained on SNLI.

### Running the trained models from the paper or viewing their output

First, download the log and checkpoint package from [here](http://nyu.edu/projects/bowman/williams_syntax_checkpoints.zip) (warning: 4GB).

To use the trained models, run any of the commands in [`scripts/train_all_models_x5.sh`](https://github.com/nyu-mll/spinn/blob/is-it-syntax-release/scripts/train_all_models_x5.sh) with the following flags appended to the end of the command. This will evaluate the model on the specified datasets, write the resulting trees to one `.report` file per dataset, and exit.

    ... --eval_data_path ../multinli_0.9/multinli_0.9_dev_matched.jsonl:../ptb.jsonl --expanded_eval_only_mode --write_eval_report

If you simply need the trees produced by the trained models for MultiNLI (dev_matched), those can be found in the checkpoint package as `.report` files.

Run the notebook [`notebooks/generate_random_binary_parses.ipynb`](https://github.com/nyu-mll/spinn/blob/is-it-syntax-release/scripts/generate_random_binary_parses.ipynb) to generate the random parse trees used by some of the baseline experiments.

### License

Copyright 2017, New York University

Licensed under the Apache License, Version 2.0 (the "License");
you may not use these files except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

[1]: http://arxiv.org/abs/1603.06021
[2]: https://github.com/stanfordnlp/spinn/blob/master/requirements.txt
[3]: https://github.com/hans/theano-hacked/tree/8964f10e44bcd7f21ae74ea7cdc3682cc7d3258e
[4]: https://github.com/google/googletest
[5]: https://github.com/oir/deep-recursive
[6]: https://github.com/stanfordnlp/spinn/blob/5d4257f4cd15cf7213d2ff87f6f3d7f6716e2ea1/cpp/bin/stacktest.cc#L33
[7]: https://github.com/stanfordnlp/spinn/releases/tag/ACL2016
[8]: http://nlp.stanford.edu/blog/hybrid-tree-sequence-neural-networks-with-spinn/
[9]: https://github.com/stanfordnlp/spinn
[10]: https://github.com/nyu-mll/spinn/blob/master/scripts/make_listops_catalan_sweep.py
[11]: https://arxiv.org/pdf/1709.01121.pdf
