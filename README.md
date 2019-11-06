# Imitation Learning for Unsupervised Parsing
This release contains the code used for paper [An Imitation Learning Approach to Unsupervised Parsing](https://arxiv.org/pdf/1906.02276.pdf)  
Code for PRPN and Gumbel Tree-LSTM is borrowed from [PRPN codebase](https://github.com/yikangshen/PRPN) and [NYU's implementation](https://github.com/nyu-mll/spinn/tree/is-it-syntax-release) respectively.
### Preparation
Requirements:  
- Python 2.7.5
- Pytorch 0.3.1

Data to download:  
- 840B word vectors 300D: [GloVe](http://nlp.stanford.edu/projects/glove/)
- All-NLI: [MultiNLI](http://www.nyu.edu/projects/bowman/multinli/) and [SNLI](http://nlp.stanford.edu/projects/snli/)

### Running
1. Train PRPN using its original code base  
2. Forward PRPN model on All-NLI data to get predicted trees:
    `./python/prpn_util/generate_distance.sh`  
3. Shuffle the generated training set and then split it to `train/dev` set  
    `cat train.json | shuf > train_shuffled.json`  
    `head -n TRAIN_NUM train_shuffled.json > train_shuffled_train.json`  
    `tail -n VALID_NUM train_shuffled.json > train_shuffled_valid.json`
4. Step-by-step supervised learning  
    `./python/sl_rl.sh`     
5. Policy refinement  
    `./python/rl_ft.sh`