import sys
import os
import json
import math
import random
import time

import gflags
import numpy as np

from spinn import util
from spinn.data.nli import load_nli_data

from spinn.util.blocks import ModelTrainer, bundle
from spinn.util.blocks import EncodeGRU, IntraAttention, Linear, ReduceTreeGRU, ReduceTreeLSTM
from spinn.util.misc import Args
from spinn.util.logparse import parse_flags

import spinn.rl_spinn
import spinn.spinn_core_model
import spinn.plain_rnn
import spinn.cbow
import spinn.choi_pyramid

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from functools import reduce


FLAGS = gflags.FLAGS


def sequential_only():
    return FLAGS.model_type == "RNN" or FLAGS.model_type == "CBOW" or FLAGS.model_type == "ChoiPyramid"


def pad_from_left():
    return FLAGS.model_type == "RNN" or FLAGS.model_type == "CBOW"


def log_path(FLAGS, load=False):
    lp = FLAGS.load_log_path if load else FLAGS.log_path
    en = FLAGS.load_experiment_name if load else FLAGS.experiment_name
    return os.path.join(lp, en) + ".log"


def get_batch(batch, sample_num = 1):

    batch = batch

    X_batch, transitions_batch, y_batch, num_transitions_batch, example_ids, dist, silver_tree = batch
    # Truncate each batch to max length within the batch.
    X_batch_is_left_padded = pad_from_left()    # False for gumbel tree
    transitions_batch_is_left_padded = True
    max_length = np.max(num_transitions_batch)
    seq_length = X_batch.shape[1]

    #print max_length, transitions_batch

    # Truncate batch.
    X_batch = truncate(X_batch, seq_length, max_length, X_batch_is_left_padded)
    transitions_batch = truncate(transitions_batch, seq_length,
                                 max_length, transitions_batch_is_left_padded)

    if dist is not None:
        dist = truncate(dist, seq_length, max_length, X_batch_is_left_padded)

    # print X_batch.shape # <batch_size x max_length x 2>
    # print transitions_batch.shape   # <batch_size x 0 x 0> invalid for gumbel tree
    # print y_batch.shape # <batch_size>
    # print num_transitions_batch.shape   # <batch_size x 2>
    # print example_ids.shape # <batch_size>
    # print dist.shape    # <batch_size x max_length x 2>

    # repeat for multiple sample
    if sample_num > 1:
        X_batch               = np.repeat(X_batch, sample_num, axis=0)
        transitions_batch     = np.repeat(transitions_batch, sample_num, axis=0)
        y_batch               = np.repeat(y_batch, sample_num, axis=0)
        num_transitions_batch = np.repeat(num_transitions_batch, sample_num, axis=0)
        example_ids           = np.repeat(example_ids, sample_num, axis=0)
        if dist is not None:
            dist                  = np.repeat(dist, sample_num, axis=0)
        silver_tree           = np.repeat(silver_tree, sample_num, axis=0)

    return X_batch, transitions_batch, y_batch, num_transitions_batch, example_ids, dist, silver_tree


def truncate(data, seq_length, max_length, left_padded):
    if left_padded:
        data = data[:, seq_length - max_length:]
    else:
        data = data[:, :max_length]
    return data


def get_data_manager(data_type):
    # Select data format.
    # if data_type == "bl":
    #     data_manager = load_boolean_data
    # elif data_type == "sst":
    #     data_manager = load_sst_data
    # elif data_type == "sst-binary":
    #     data_manager = load_sst_binary_data
    # elif data_type == "nli":
    #     data_manager = load_nli_data
    # elif data_type == "arithmetic":
    #     data_manager = load_simple_data
    # elif data_type == "listops":
    #     data_manager = load_listops_data
    # elif data_type == "sign":
    #     data_manager = load_sign_data
    # elif data_type == "eq":
    #     data_manager = load_eq_data
    # elif data_type == "relational":
    #     data_manager = load_relational_data
    # else:
    #     raise NotImplementedError
    
    if data_type == "nli":
        data_manager = load_nli_data
    else:
        raise NotImplementedError

    return data_manager

def get_checkpoint_path_for_sl(
        ckpt_path,  # directory
        experiment_name,
        suffix=".ckpt",
        step=0):
        return os.path.join(ckpt_path, experiment_name + '_sl-step-{:d}'.format(step) + suffix)

def get_checkpoint_path(
        ckpt_path,  # directory
        experiment_name,
        suffix=".ckpt",
        best=False,
        parsing=False):
    # Set checkpoint path.

    if FLAGS.expanded_eval_only_mode and FLAGS.expanded_eval_only_mode_use_best_checkpoint:
        best = True

    if ckpt_path.endswith(".ckpt") or ckpt_path.endswith(".ckpt_best") or ckpt_path.endswith(".ckpt_best_parsing"):
        checkpoint_path = ckpt_path
    else:
        checkpoint_path = os.path.join(ckpt_path, experiment_name + suffix)
    if best:
        checkpoint_path += "_best"
        if parsing:
            checkpoint_path += "_parsing"
    return checkpoint_path


def load_data_and_embeddings(
        FLAGS,
        data_manager,
        logger,
        training_data_path,
        eval_data_path):

    def choose_train(x): return True

    # BOON
    # if FLAGS.train_genre is not None:
    #     def choose_train(x): return x.get('genre') == FLAGS.train_genre

    def choose_eval(x): return True

    # BOON
    # if FLAGS.eval_genre is not None:
    #     def choose_eval(x): return x.get('genre') == FLAGS.eval_genre

    if not FLAGS.expanded_eval_only_mode:
        if FLAGS.data_type == "nli":
            # Load the data.
            raw_training_data = data_manager.load_data(
                training_data_path, FLAGS.lowercase, choose_train, mode=FLAGS.transition_mode, 
                tree_joint=FLAGS.tree_joint,
                distance_type=FLAGS.distance_type)    # only for nli data now
        else:
            # Load the data.
            raw_training_data = data_manager.load_data(
                training_data_path, FLAGS.lowercase, mode=FLAGS.transition_mode)
    else:
        raw_training_data = None

    if FLAGS.data_type == "nli":
        # Load the eval data.
        raw_eval_sets = []
        for path in eval_data_path.split(':'):
            raw_eval_data = data_manager.load_data(
                path, FLAGS.lowercase, choose_eval, mode=FLAGS.transition_mode,
                tree_joint=FLAGS.tree_joint,
                distance_type=FLAGS.distance_type)    # only for nli data now
            raw_eval_sets.append((path, raw_eval_data))
            # print raw_eval_data[1].keys()
            #exit(1)
    else:
        # Load the eval data.
        raw_eval_sets = []
        for path in eval_data_path.split(':'):
            raw_eval_data = data_manager.load_data(path, FLAGS.lowercase, mode=FLAGS.transition_mode)
            raw_eval_sets.append((path, raw_eval_data))

    # Prepare the vocabulary.
    if not data_manager.FIXED_VOCABULARY:
        logger.Log(
            "In open vocabulary mode. Using loaded embeddings without fine-tuning.")
        vocabulary = util.BuildVocabulary(
            raw_training_data,
            raw_eval_sets,
            FLAGS.embedding_data_path,
            logger=logger,
            sentence_pair_data=data_manager.SENTENCE_PAIR_DATA)
    else:
        vocabulary = data_manager.FIXED_VOCABULARY
        logger.Log("In fixed vocabulary mode. Training embeddings.")

    # Load pretrained embeddings.
    if FLAGS.embedding_data_path:
        logger.Log("Loading vocabulary with " + str(len(vocabulary))
                   + " words from " + FLAGS.embedding_data_path)
        initial_embeddings = util.LoadEmbeddingsFromText(
            vocabulary, FLAGS.word_embedding_dim, FLAGS.embedding_data_path)
    else:
        initial_embeddings = None

    # Trim dataset, convert token sequences to integer sequences, crop, and
    # pad.
    logger.Log("Preprocessing training data.")
    training_data = util.PreprocessDataset(
        raw_training_data,
        vocabulary,
        FLAGS.seq_length,
        data_manager,
        eval_mode=False,
        logger=logger,
        sentence_pair_data=data_manager.SENTENCE_PAIR_DATA,
        simple=sequential_only(),
        allow_cropping=FLAGS.allow_cropping,
        pad_from_left=pad_from_left(),
        tree_joint=FLAGS.tree_joint) if raw_training_data is not None else None
    training_data_iter = util.MakeTrainingIterator(
        training_data, FLAGS.batch_size, FLAGS.smart_batching, FLAGS.use_peano,
        sentence_pair_data=data_manager.SENTENCE_PAIR_DATA
        # ,train_seed=FLAGS.train_seed
        ) if raw_training_data is not None else None

    # Preprocess eval sets.
    eval_iterators = []
    for filename, raw_eval_set in raw_eval_sets:
        logger.Log("Preprocessing eval data: " + filename)
        eval_data = util.PreprocessDataset(
            raw_eval_set, vocabulary,
            FLAGS.eval_seq_length if FLAGS.eval_seq_length is not None else FLAGS.seq_length,
            data_manager, eval_mode=True, logger=logger,
            sentence_pair_data=data_manager.SENTENCE_PAIR_DATA,
            simple=sequential_only(),
            allow_cropping=FLAGS.allow_eval_cropping, 
            pad_from_left=pad_from_left(),
            tree_joint=FLAGS.tree_joint)

        eval_it = util.MakeEvalIterator(
            eval_data,
            FLAGS.batch_size * FLAGS.sample_num,    # keep the eval running speed
            FLAGS.eval_data_limit,
            bucket_eval=FLAGS.bucket_eval,
            shuffle=FLAGS.shuffle_eval,
            rseed=FLAGS.shuffle_eval_seed)
        eval_iterators.append((filename, eval_it))
    return vocabulary, initial_embeddings, training_data_iter, eval_iterators


def get_flags():
    # Debug settings.
    gflags.DEFINE_bool(
        "debug",
        False,
        "Set to True to disable debug_mode and type_checking.")
    gflags.DEFINE_bool(
        "show_progress_bar",
        True,
        "Turn this off when running experiments on HPC.")
    gflags.DEFINE_string("git_branch_name", "", "Set automatically.")
    gflags.DEFINE_string("slurm_job_id", "", "Set automatically.")
    gflags.DEFINE_integer(
        "deque_length",
        100,
        "Max trailing examples to use when computing average training statistics.")
    gflags.DEFINE_string("git_sha", "", "Set automatically.")
    gflags.DEFINE_string("experiment_name", "", "")
    gflags.DEFINE_string("load_experiment_name", None, "")

    # Data types.
    gflags.DEFINE_enum("data_type",
                       "bl",
                       ["bl",
                        "sst",
                        "sst-binary",
                        "nli",
                        "arithmetic",
                        "listops",
                        "sign",
                        "eq",
                        "relational"],
                       "Which data handler and classifier to use.")

    # Choose Genre.
    # 'fiction', 'government', 'slate', 'telephone', 'travel'
    # 'facetoface', 'letters', 'nineeleven', 'oup', 'verbatim'
    gflags.DEFINE_string("train_genre", None, "Filter MultiNLI data by genre.")
    gflags.DEFINE_string("eval_genre", None, "Filter MultiNLI data by genre.")

    # Where to store checkpoints
    gflags.DEFINE_string(
        "log_path",
        "./logs",
        "A directory in which to write logs.")
    gflags.DEFINE_string(
        "load_log_path",
        None,
        "A directory in which to write logs.")
    gflags.DEFINE_boolean(
        "write_proto_to_log",
        False,
        "Write logs in a protocol buffer format.")
    gflags.DEFINE_string(
        "ckpt_path", None, "Where to save/load checkpoints. Can be either "
        "a filename or a directory. In the latter case, the experiment name serves as the "
        "base for the filename.")
    gflags.DEFINE_string(
        "metrics_path",
        None,
        "A directory in which to write metrics.")
    gflags.DEFINE_integer(
        "ckpt_step",
        1000,
        "Steps to run before considering saving checkpoint.")
    gflags.DEFINE_boolean(
        "load_best",
        False,
        "If True, attempt to load 'best' checkpoint.")

    # Data settings.
    gflags.DEFINE_string("training_data_path", None, "")
    gflags.DEFINE_string(
        "eval_data_path", None, "Can contain multiple file paths, separated "
        "using ':' tokens. The first file should be the dev set, and is used for determining "
        "when to save the early stopping 'best' checkpoints.")
    gflags.DEFINE_integer("seq_length", 200, "")
    gflags.DEFINE_boolean(
        "allow_cropping",
        False,
        "Trim overly long training examples to fit. If not set, skip them.")
    gflags.DEFINE_integer("eval_seq_length", None, "")
    gflags.DEFINE_boolean(
        "allow_eval_cropping",
        False,
        "Trim overly long evaluation examples to fit. If not set, crash on overly long examples.")
    gflags.DEFINE_boolean(
        "smart_batching",
        True,
        "Organize batches using sequence length.")
    gflags.DEFINE_boolean("use_peano", True, "A mind-blowing sorting key.")
    gflags.DEFINE_integer(
        "eval_data_limit",
        None,
        "Truncate evaluation set to this many batches. -1 indicates no truncation.")
    gflags.DEFINE_boolean(
        "bucket_eval",
        False,
        "Bucket evaluation data for speed improvement.")
    gflags.DEFINE_boolean("shuffle_eval", False, "Shuffle evaluation data.")
    gflags.DEFINE_integer(
        "shuffle_eval_seed",
        123,
        "Seed shuffling of eval data.")
    gflags.DEFINE_string("embedding_data_path", None,
                         "If set, load GloVe-formatted embeddings from here.")

    # Model architecture settings.
    gflags.DEFINE_enum(
        "model_type", "RNN", [
            "CBOW", "RNN", "SPINN", "RLSPINN", "ChoiPyramid"], "")
    gflags.DEFINE_integer("gpu", -1, "")
    gflags.DEFINE_integer("model_dim", 8, "")
    gflags.DEFINE_integer("word_embedding_dim", 8, "")
    gflags.DEFINE_boolean("lowercase", False, "When True, ignore case.")
    gflags.DEFINE_boolean("use_internal_parser", False, "Use predicted parse.")
    gflags.DEFINE_boolean(
        "validate_transitions",
        True,
        "Constrain predicted transitions to ones that give a valid parse tree.")
    gflags.DEFINE_float(
        "embedding_keep_rate",
        0.9,
        "Used for dropout on transformed embeddings and in the encoder RNN.")
    gflags.DEFINE_boolean("use_l2_loss", True, "")
    gflags.DEFINE_boolean("use_difference_feature", True, "")
    gflags.DEFINE_boolean("use_product_feature", True, "")

    # Tracker settings.
    gflags.DEFINE_integer(
        "tracking_lstm_hidden_dim",
        None,
        "Set to none to avoid using tracker.")
    gflags.DEFINE_boolean(
        "tracking_ln",
        False,
        "When True, layer normalization is used in tracking.")
    gflags.DEFINE_float(
        "transition_weight",
        None,
        "Set to none to avoid predicting transitions.")
    gflags.DEFINE_boolean("lateral_tracking", True,
                          "Use previous tracker state as input for new state.")
    gflags.DEFINE_boolean(
        "use_tracking_in_composition",
        True,
        "Use tracking lstm output as input for the reduce function.")
    gflags.DEFINE_boolean(
        "composition_ln",
        True,
        "When True, layer normalization is used in TreeLSTM composition.")
    gflags.DEFINE_boolean("predict_use_cell", True,
                          "Use cell output as feature for transition net.")

    # Reduce settings.
    gflags.DEFINE_enum(
        "reduce", "treelstm", [
            "treelstm", "treegru", "tanh"], "Specify composition function.")

    # Fixed tree settings.
    gflags.DEFINE_enum(
        "transition_mode", "default", [
            "default", "full", "half_full", "half_full_right", "balanced"],
            "Specify whether to use given" +\
            " binary parse trees or to use a fixed strategy.")
    gflags.DEFINE_boolean(
        "full_trees",
        False,
        "If set to True, then all parse trees will be full binary" +\
        " trees and sentences padded to a factor of 2. (deprecated;" +\
        " use transition_mode instead)")

    # Pyramid model settings
    gflags.DEFINE_boolean(
        "pyramid_trainable_temperature",
        None,
        "If set, add a scalar trained temperature parameter.")
    gflags.DEFINE_float("pyramid_temperature_decay_per_10k_steps",
                        0.5, "What it says on the box.")
    gflags.DEFINE_float(
        "pyramid_temperature_cycle_length",
        0.0,
        "For wake-sleep-style experiments. 0.0 disables this feature.")

    # Encode settings.
    gflags.DEFINE_enum("encode",
                       "projection",
                       ["pass",
                        "projection",
                        "gru",
                        "attn"],
                       "Encode embeddings with sequential context.")
    gflags.DEFINE_boolean("encode_reverse", False, "Encode in reverse order.")
    gflags.DEFINE_boolean(
        "encode_bidirectional",
        False,
        "Encode in both directions.")
    gflags.DEFINE_integer(
        "encode_num_layers",
        1,
        "RNN layers in encoding net.")

    # RL settings.
    gflags.DEFINE_float(
        "rl_mu",
        0.1,
        "Use in exponential moving average baseline.")
    gflags.DEFINE_enum("rl_baseline",
                       "ema",
                       ["ema",
                        "pass",
                        "greedy",
                        "value"],
                       "Different configurations to approximate reward function.")
    gflags.DEFINE_enum("rl_reward", "standard", ["standard", "xent"],
                       "Different reward functions to use.")
    gflags.DEFINE_float("rl_weight", 1.0, "Hyperparam for REINFORCE loss.")
    gflags.DEFINE_boolean("rl_whiten", False, "Reduce variance in advantage.")
    gflags.DEFINE_boolean(
        "rl_valid",
        True,
        "Only consider non-validated actions.")
    gflags.DEFINE_float(
        "rl_epsilon",
        1.0,
        "Percent of sampled actions during train time.")
    gflags.DEFINE_float(
        "rl_epsilon_decay",
        50000,
        "Step constant in epsilon delay equation.")
    gflags.DEFINE_float(
        "rl_confidence_interval",
        1000,
        "Penalize probabilities of transitions.")
    gflags.DEFINE_float(
        "rl_confidence_penalty",
        None,
        "Penalize probabilities of transitions.")
    gflags.DEFINE_boolean(
        "rl_catalan",
        False,
        "Sample over a uniform distribution of binary trees.")
    gflags.DEFINE_boolean(
        "rl_catalan_backprop",
        False,
        "Sample over a uniform distribution of binary trees.")
    gflags.DEFINE_boolean(
        "rl_wake_sleep",
        False,
        "Inverse relationship between temperature and rl_weight.")
    gflags.DEFINE_boolean(
        "rl_transition_acc_as_reward",
        False,
        "Use the transition accuracy as the reward. For debugging only.")

    # RNN Settings
    gflags.DEFINE_boolean(
        "model_bidirectional",
        False,
        "Use bidirectional recurrent network.")

    # MLP settings.
    gflags.DEFINE_integer(
        "mlp_dim",
        1024,
        "Dimension of intermediate MLP layers.")
    gflags.DEFINE_integer("num_mlp_layers", 2, "Number of MLP layers.")
    gflags.DEFINE_boolean(
        "mlp_ln",
        True,
        "When True, layer normalization is used between MLP layers.")
    gflags.DEFINE_float("semantic_classifier_keep_rate", 0.9,
                        "Used for dropout in the semantic task classifier.")

    # Optimization settings.
    gflags.DEFINE_enum(
        "optimizer_type", "Adam", [
            "Adam", "RMSprop", "YellowFin"], "")
    gflags.DEFINE_integer(
        "training_steps",
        500000,
        "Stop training after this point.")
    gflags.DEFINE_integer("batch_size", 32, "SGD minibatch size.")
    gflags.DEFINE_float("learning_rate", 0.001, "Used in optimizer.")
    gflags.DEFINE_float(
        "learning_rate_decay_per_10k_steps",
        0.75,
        "Used in optimizer.")
    gflags.DEFINE_boolean(
        "actively_decay_learning_rate",
        True,
        "Used in optimizer.")
    gflags.DEFINE_float("clipping_max_value", 5.0, "")
    gflags.DEFINE_float("l2_lambda", 1e-5, "")
    gflags.DEFINE_float(
        "init_range",
        0.005,
        "Mainly used for softmax parameters. Range for uniform random init.")

    # Display settings.
    gflags.DEFINE_integer(
        "statistics_interval_steps",
        100,
        "Log training set performance statistics at this interval.")
    gflags.DEFINE_integer(
        "eval_interval_steps",
        100,
        "Evaluate at this interval.")
    gflags.DEFINE_integer(
        "sample_interval_steps",
        None,
        "Sample transitions at this interval.")
    gflags.DEFINE_integer("ckpt_interval_steps", 5000,
                          "Update the checkpoint on disk at this interval.")
    gflags.DEFINE_boolean(
        "ckpt_on_best_dev_error",
        True,
        "If error on the first eval set (the dev set) is "
        "at most 0.99 of error at the previous checkpoint, save a special 'best' checkpoint.")
    gflags.DEFINE_integer(
        "early_stopping_steps_to_wait",
        25000,
        "If development set error doesn't improve significantly in this many steps, cease training.")
    gflags.DEFINE_boolean("evalb", False, "Print transition statistics.")
    gflags.DEFINE_integer("num_samples", 0, "Print sampled transitions.")

    # Evaluation settings
    gflags.DEFINE_boolean(
        "expanded_eval_only_mode",
        False,
        "If set, a checkpoint is loaded and a forward pass is done to get the predicted "
        "transitions. The inferred parses are written to the supplied file(s) along with example-"
        "by-example accuracy information. Requirements: Must specify checkpoint path.")  # TODO: Rename.
    gflags.DEFINE_boolean(
        "expanded_eval_only_mode_use_best_checkpoint",
        True,
        "When in expanded_eval_only_mode, load the ckpt_best checkpoint.")
    gflags.DEFINE_boolean("write_eval_report", False, "")
    gflags.DEFINE_boolean(
        "eval_report_use_preds", True, "If False, use the given transitions in the report, "
        "otherwise use predicted transitions. Note that when predicting transitions but not using them, the "
        "reported predictions will look very odd / not valid.")  # TODO: Remove.

    # Evolution Strategy
    gflags.DEFINE_boolean(
        "transition_detach",
        False,
        "Detach transition decision from backprop.")
    gflags.DEFINE_boolean("evolution", False, "Use evolution to train parser.")
    gflags.DEFINE_float(
        "es_sigma",
        0.05,
        "Standard deviation for Gaussian noise.")
    gflags.DEFINE_integer(
        "es_num_episodes",
        2,
        "Number of simultaneous episodes to run.")
    gflags.DEFINE_integer(
        "es_num_roots",
        2,
        "Number of simultaneous episodes to run.")
    gflags.DEFINE_integer("es_episode_length", 1000, "Length of each episode.")
    gflags.DEFINE_integer("es_steps", 1000, "Number of evolution steps.")
    gflags.DEFINE_boolean(
        "mirror",
        False,
        "Do mirrored/antithetic sampling. If doing mirrored sampling, number of perturbtations will be double es_num_episodes.")
    gflags.DEFINE_float(
        "eval_sample_size",
        None,
        "Percentage (eg 0.3) of batches to be sampled for evaluation during training (only for ES). If None, use all.")
    # gflags.DEFINE_string("dist_path", None, "using distance")
    gflags.DEFINE_boolean(
        "tree_joint",
        False,
        "Train the classifier jointly with the syntactical distance target. \
        If it is True, use the processed nli dataset which is different from the original nli dataset especially in tree parse.")
    gflags.DEFINE_enum(
        "distance_type", "definition", ["definition", "right-most"], "")
    gflags.DEFINE_integer(
        "sbs_step",
        500000,
        "# of steps of step-by-step training. 0=no sbs")
    gflags.DEFINE_float(
        "sbs_weight",
        0.1,
        "weight for sbs loss.")
    gflags.DEFINE_enum(
        "test_type", "classification", ["classification", "parsing"], "early stop by classification or parsing")
    gflags.DEFINE_boolean(
       "save_sl",
       False,
       "save sl")
    gflags.DEFINE_integer(
        "save_sl_step",
        1000,
        "# of steps to save sl checkpoints, mush set save_sl True first")
    gflags.DEFINE_boolean("continue_train", False, "whether or not load the optimizer state when loading a checkpoint")
    gflags.DEFINE_boolean("load_sl", False, "load a pretrained sl checkpoint instead of a standard checkpoint")
    gflags.DEFINE_integer(
        "load_sl_step",
        1000,
        "# of steps to load sl checkpoints")
    gflags.DEFINE_boolean("customize_ckpt", False, "load a customized pretrained checkpoint, must be used with customized_ckpt_path")    
    gflags.DEFINE_string(
        "customize_ckpt_path", None, "full path for customized pretrained checkpoint")
    gflags.DEFINE_integer(
        "sample_num",
        1,
        "Number of action sequences to sample. Data batch will be batch size / sample_num.")
    gflags.DEFINE_string("prpn_name", "", "PRPN experiment name")


def flag_defaults(FLAGS, load_log_flags=False):
    if load_log_flags:
        if FLAGS.load_log_path and os.path.exists(log_path(FLAGS, load=True)):
            log_flags = parse_flags(log_path(FLAGS, load=True))
            for k in log_flags.keys():
                setattr(FLAGS, k, log_flags[k])

            # Optionally override flags from log file.
            FLAGS(sys.argv)

    if not FLAGS.experiment_name:
        timestamp = str(int(time.time()))
        FLAGS.experiment_name = "{}-{}-{}".format(
            FLAGS.data_type,
            FLAGS.model_type,
            timestamp,
        )

    if not FLAGS.git_branch_name:
        FLAGS.git_branch_name = os.popen(
            'git rev-parse --abbrev-ref HEAD').read().strip()

    if not FLAGS.git_sha:
        FLAGS.git_sha = os.popen('git rev-parse HEAD').read().strip()

    if not FLAGS.slurm_job_id:
        FLAGS.slurm_job_id = os.popen('echo $SLURM_JOB_ID').read().strip()

    if not FLAGS.load_log_path:
        FLAGS.load_log_path = FLAGS.log_path

    if not FLAGS.load_experiment_name:
        FLAGS.load_experiment_name = FLAGS.experiment_name

    if not FLAGS.ckpt_path:
        FLAGS.ckpt_path = FLAGS.load_log_path

    if not FLAGS.sample_interval_steps:
        FLAGS.sample_interval_steps = FLAGS.statistics_interval_steps

    if not FLAGS.metrics_path:
        FLAGS.metrics_path = FLAGS.log_path

    if FLAGS.model_type == "CBOW" or FLAGS.model_type == "RNN" or FLAGS.model_type == "Pyramid" or FLAGS.model_type == "ChoiPyramid":
        FLAGS.num_samples = 0

    if not torch.cuda.is_available():
        FLAGS.gpu = -1

    if FLAGS.full_trees:
        print('Using deprecated flag full_trees. Use transition_mode instead.')
        assert FLAGS.transition_mode == 'default', 'If full_trees is set, then do not use transition_mode.'
        FLAGS.transition_mode = 'full'


def init_model(
        FLAGS,
        logger,
        initial_embeddings,
        vocab_size,
        num_classes,
        data_manager,
        logfile_header=None):
    # Choose model.
    logger.Log("Building model.")
    if FLAGS.model_type == "CBOW":
        build_model = spinn.cbow.build_model
    elif FLAGS.model_type == "RNN":
        build_model = spinn.plain_rnn.build_model
    elif FLAGS.model_type == "SPINN":
        build_model = spinn.spinn_core_model.build_model
    elif FLAGS.model_type == "RLSPINN":
        build_model = spinn.rl_spinn.build_model
    elif FLAGS.model_type == "ChoiPyramid":
        build_model = spinn.choi_pyramid.build_model
    else:
        raise NotImplementedError


    # Input Encoder.
    context_args = Args()
    context_args.reshape_input = lambda x, batch_size, seq_length: x
    context_args.reshape_context = lambda x, batch_size, seq_length: x
    context_args.input_dim = FLAGS.word_embedding_dim

    if FLAGS.encode == "projection":
        encoder = Linear()(FLAGS.word_embedding_dim, FLAGS.model_dim)
        context_args.input_dim = FLAGS.model_dim
    elif FLAGS.encode == "gru":
        context_args.reshape_input = lambda x, batch_size, seq_length: x.view(
            batch_size, seq_length, -1)
        context_args.reshape_context = lambda x, batch_size, seq_length: x.view(
            batch_size * seq_length, -1)
        context_args.input_dim = FLAGS.model_dim
        encoder = EncodeGRU(FLAGS.word_embedding_dim, FLAGS.model_dim,
                            num_layers=FLAGS.encode_num_layers,
                            bidirectional=FLAGS.encode_bidirectional,
                            reverse=FLAGS.encode_reverse,
                            mix=(FLAGS.model_type != "CBOW"))
    elif FLAGS.encode == "attn":
        context_args.reshape_input = lambda x, batch_size, seq_length: x.view(
            batch_size, seq_length, -1)
        context_args.reshape_context = lambda x, batch_size, seq_length: x.view(
            batch_size * seq_length, -1)
        context_args.input_dim = FLAGS.model_dim
        encoder = IntraAttention(FLAGS.word_embedding_dim, FLAGS.model_dim)
    elif FLAGS.encode == "pass":
        def encoder(x): return x
    else:
        raise NotImplementedError

    context_args.encoder = encoder

    # Composition Function.
    composition_args = Args()
    composition_args.lateral_tracking = FLAGS.lateral_tracking
    composition_args.tracking_ln = FLAGS.tracking_ln
    composition_args.use_tracking_in_composition = FLAGS.use_tracking_in_composition
    composition_args.size = FLAGS.model_dim
    composition_args.tracker_size = FLAGS.tracking_lstm_hidden_dim
    composition_args.use_internal_parser = FLAGS.use_internal_parser
    composition_args.transition_weight = FLAGS.transition_weight
    composition_args.wrap_items = lambda x: torch.cat(x, 0)
    composition_args.extract_h = lambda x: x

    composition_args.detach = FLAGS.transition_detach
    composition_args.evolution = FLAGS.evolution

    if FLAGS.reduce == "treelstm":
        assert FLAGS.model_dim % 2 == 0, 'model_dim must be an even number.'
        if FLAGS.model_dim != FLAGS.word_embedding_dim:
            print('If you are setting different hidden layer and word '
                  'embedding sizes, make sure you specify an encoder')
        composition_args.wrap_items = lambda x: bundle(x)
        composition_args.extract_h = lambda x: x.h
        composition_args.extract_c = lambda x: x.c
        composition_args.size = FLAGS.model_dim / 2
        composition = ReduceTreeLSTM(
            FLAGS.model_dim / 2,
            tracker_size=FLAGS.tracking_lstm_hidden_dim,
            use_tracking_in_composition=FLAGS.use_tracking_in_composition,
            composition_ln=FLAGS.composition_ln)
    elif FLAGS.reduce == "tanh":
        class ReduceTanh(nn.Module):
            def forward(self, lefts, rights, tracking=None):
                batch_size = len(lefts)
                ret = torch.cat(lefts, 0) + F.tanh(torch.cat(rights, 0))
                return torch.chunk(ret, batch_size, 0)
        composition = ReduceTanh()
    elif FLAGS.reduce == "treegru":
        composition = ReduceTreeGRU(FLAGS.model_dim,
                                    FLAGS.tracking_lstm_hidden_dim,
                                    FLAGS.use_tracking_in_composition)
    else:
        raise NotImplementedError

    composition_args.composition = composition

    model = build_model(data_manager, initial_embeddings, vocab_size,
                        num_classes, FLAGS, context_args, composition_args)

    # Build optimizer.
    if FLAGS.optimizer_type == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=FLAGS.learning_rate,
                               betas=(0.9, 0.999), eps=1e-08)
    elif FLAGS.optimizer_type == "RMSprop":
        optimizer = optim.RMSprop(
            model.parameters(),
            lr=FLAGS.learning_rate,
            eps=1e-08)
    elif FLAGS.optimizer_type == "YellowFin":
        raise NotImplementedError
    else:
        raise NotImplementedError

    # Build trainer.
    if FLAGS.evolution:
        # BOON
        # trainer = ModelTrainer_ES(model, optimizer)
        raise NotImplementedError
    else:
        trainer = ModelTrainer(model, optimizer)

    # Print model size.
    logger.Log("Architecture: {}".format(model))

    if logfile_header:
        logfile_header.model_architecture = str(model)
    total_params = sum([reduce(lambda x, y: x * y, w.size(), 1.0)
                        for w in model.parameters()])
    logger.Log("Total params: {}".format(total_params))
    if logfile_header:
        logfile_header.total_params = int(total_params)

    return model, optimizer, trainer
