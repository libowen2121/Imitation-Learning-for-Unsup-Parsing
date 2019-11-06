import os
import json
import random
import sys
import time
import math

import gflags
import numpy as np

random.seed(310) # for reproducing purpose

from spinn.util import afs_safe_logger
from spinn.util.data import SimpleProgressBar, get_brackets, convert_brackets_to_string
from spinn.util.blocks import get_l2_loss, the_gpu, to_gpu, DefaultUniformInitializer
from spinn.util.misc import Accumulator, EvalReporter
from spinn.util.misc import recursively_set_device
from spinn.util.logging import stats, train_accumulate, create_log_formatter
from spinn.util.logging import stats, train_accumulate, create_log_formatter
from spinn.util.logging import eval_stats, eval_accumulate, prettyprint_trees
from spinn.util.loss import auxiliary_loss
from spinn.util.sparks import sparks, dec_str
import spinn.util.evalb as evalb
import spinn.util.logging_pb2 as pb

# PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


from spinn.models.base import get_data_manager, get_flags, get_batch
from spinn.models.base import flag_defaults, init_model
from spinn.models.base import get_checkpoint_path, log_path, get_checkpoint_path_for_sl
from spinn.models.base import load_data_and_embeddings

import cPickle
import os.path

FLAGS = gflags.FLAGS


def evaluate(FLAGS, model, data_manager, eval_set, log_entry,
             logger, step, vocabulary=None, show_sample=False, eval_index=0):
    filename, dataset = eval_set

    A = Accumulator()
    index = len(log_entry.evaluation)
    eval_log = log_entry.evaluation.add()
    reporter = EvalReporter()
    tree_strs = None

    # Evaluate
    total_batches = len(dataset)
    progress_bar = SimpleProgressBar(
        msg="Run Eval",
        bar_length=60,
        enabled=FLAGS.show_progress_bar)
    progress_bar.step(0, total=total_batches)
    total_tokens = 0
    start = time.time()

    if FLAGS.model_type in ["Pyramid", "ChoiPyramid"]:
        pyramid_temperature_multiplier = FLAGS.pyramid_temperature_decay_per_10k_steps ** (
            step / 10000.0)
        if FLAGS.pyramid_temperature_cycle_length > 0.0:
            min_temp = 1e-5
            pyramid_temperature_multiplier *= (
                math.cos((step) / FLAGS.pyramid_temperature_cycle_length) + 1 + min_temp) / 2
    else:
        pyramid_temperature_multiplier = None

    model.eval()

    for i, dataset_batch in enumerate(dataset):
        batch = get_batch(dataset_batch)
        eval_X_batch, eval_transitions_batch, eval_y_batch, eval_num_transitions_batch, eval_ids, _, silver_tree = batch
            # eval_X_batch: <batch x maxlen x 2>
            # eval_y_batch: <batch >
            # silver_tree:
            # the dist is invalid for val 

        # Run model.
        output = model(
            eval_X_batch,
            eval_transitions_batch,
            eval_y_batch,
            use_internal_parser=FLAGS.use_internal_parser,
            validate_transitions=FLAGS.validate_transitions,
            pyramid_temperature_multiplier=pyramid_temperature_multiplier,
            store_parse_masks=True,
            example_lengths=eval_num_transitions_batch)

        # TODO: Restore support in Pyramid if using.
        can_sample = FLAGS.model_type in ["ChoiPyramid"] or (
            FLAGS.model_type == "SPINN" and FLAGS.use_internal_parser)
        if show_sample and can_sample:
            tmp_samples = model.get_samples(
                eval_X_batch, vocabulary, only_one=not FLAGS.write_eval_report)
            # tree_strs = prettyprint_trees(tmp_samples)
            tree_strs = [tree for tree in tmp_samples]
        
        tmp_samples = model.get_samples(
            eval_X_batch, vocabulary, only_one=False)
        
        # def get_max(s):
        #     # test f1
        #     max = 0
        #     for x in s:
        #         _, idx = x.split(',')
        #         if int(idx) > max:
        #             max = int(idx)
        #     return max

        for s in (range(int(model.use_sentence_pair) + 1)):
            for b in (range(silver_tree.shape[0])):
                model_out = tmp_samples[s * silver_tree.shape[0] + b]
                std_out = silver_tree[b, :, s]
                std_out = set([x for x in std_out if x != '-1,-1'])
                model_out_brackets, model_out_max_l = get_brackets(model_out)
                model_out = set(convert_brackets_to_string(model_out_brackets))
                
                outmost_bracket = '{:d},{:d}'.format(0, model_out_max_l)
                std_out.add(outmost_bracket)
                model_out.add(outmost_bracket)

                # print get_max(model_out), get_max(std_out)
                # print model_out
                # print std_out
                # print '=' * 30
                # assert get_max(model_out) == get_max(std_out)

                overlap = model_out & std_out
                prec = float(len(overlap)) / (len(model_out) + 1e-8)
                reca = float(len(overlap)) / (len(std_out) + 1e-8)
                if len(std_out) == 0:
                    reca = 1.
                    if len(model_out) == 0:
                        prec = 1.
                f1 = 2 * prec * reca / (prec + reca + 1e-8)    
                A.add('f1', f1)  

        if not FLAGS.write_eval_report:
            # Only show one sample, regardless of the number of batches.
            show_sample = False

        # Normalize output.
        logits = F.log_softmax(output)

        # Calculate class accuracy.
        target = torch.from_numpy(eval_y_batch).long()

        # get the index of the max log-probability
        pred = logits.data.max(1, keepdim=False)[1].cpu()

        eval_accumulate(model, data_manager, A, batch)
        A.add('class_correct', pred.eq(target).sum())
        A.add('class_total', target.size(0))


        # Optionally calculate transition loss/acc.
        #model.transition_loss if hasattr(model, 'transition_loss') else None
        # TODO: review this. the original line seems to have no effect

        # Update Aggregate Accuracies
        total_tokens += sum([(nt + 1) /
                             2 for nt in eval_num_transitions_batch.reshape(-1)])

        if FLAGS.write_eval_report:
            transitions_per_example, _ = model.spinn.get_transitions_per_example(
                style="preds" if FLAGS.eval_report_use_preds else "given") if (
                FLAGS.model_type == "SPINN" and FLAGS.use_internal_parser) else (
                None, None)

            if model.use_sentence_pair:
                batch_size = pred.size(0)
                sent1_transitions = transitions_per_example[:
                                                            batch_size] if transitions_per_example is not None else None
                sent2_transitions = transitions_per_example[batch_size:
                                                            ] if transitions_per_example is not None else None

                sent1_trees = tree_strs[:batch_size] if tree_strs is not None else None
                sent2_trees = tree_strs[batch_size:
                                        ] if tree_strs is not None else None
            else:
                sent1_transitions = transitions_per_example if transitions_per_example is not None else None
                sent2_transitions = None

                sent1_trees = tree_strs if tree_strs is not None else None
                sent2_trees = None

            reporter.save_batch(
                pred,
                target,
                eval_ids,
                output.data.cpu().numpy(),
                sent1_transitions,
                sent2_transitions,
                sent1_trees,
                sent2_trees)

        # Print Progress
        progress_bar.step(i + 1, total=total_batches)
    progress_bar.finish()
    if tree_strs is not None:
        logger.Log('Sample: ' + str(tree_strs[0]))

    end = time.time()
    total_time = end - start

    A.add('total_tokens', total_tokens)
    A.add('total_time', total_time)

    eval_stats(model, A, eval_log)  # get the eval statistics (e.g. average F1)
    eval_log.filename = filename

    if FLAGS.write_eval_report:
        eval_report_path = os.path.join(
            FLAGS.log_path,
            FLAGS.experiment_name +
            ".eval_set_" +
            str(eval_index) +
            ".report")
        reporter.write_report(eval_report_path)

    eval_class_acc = eval_log.eval_class_accuracy
    eval_trans_acc = eval_log.eval_transition_accuracy
    eval_f1 = eval_log.f1

    return eval_class_acc, eval_trans_acc, eval_f1


def train_loop(
        FLAGS,
        data_manager,
        model,
        optimizer,
        trainer,
        training_data_iter,
        eval_iterators,
        logger,
        step,
        best_dev_error,
        best_dev_step,
        best_dev_f1_error,
        vocabulary):
    # Accumulate useful statistics.
    A = Accumulator(maxlen=FLAGS.deque_length)

    # Checkpoint paths.
    standard_checkpoint_path = get_checkpoint_path(
        FLAGS.ckpt_path, FLAGS.experiment_name)
    best_checkpoint_path = get_checkpoint_path(
        FLAGS.ckpt_path, FLAGS.experiment_name, best=True)
    best_parsing_checkpoint_path = get_checkpoint_path(
        FLAGS.ckpt_path, FLAGS.experiment_name, best=True, parsing=True)


    model.train()

    # Train.
    logger.Log("Training.")

    # New Training Loop
    progress_bar = SimpleProgressBar(
        msg="Training",
        bar_length=60,
        enabled=FLAGS.show_progress_bar)
    progress_bar.step(i=0, total=FLAGS.statistics_interval_steps)

    log_entry = pb.SpinnEntry()
    for step in range(step, FLAGS.training_steps):
        if (step - best_dev_step) > FLAGS.early_stopping_steps_to_wait:
            logger.Log('No improvement after ' + str(FLAGS.early_stopping_steps_to_wait) + ' steps. Stopping training.')
            break

        model.train()
        log_entry.Clear()
        log_entry.step = step
        should_log = False

        start = time.time()

        batch = get_batch(training_data_iter.next(), FLAGS.sample_num)
        X_batch, transitions_batch, y_batch, num_transitions_batch, train_ids, dist_batch, _ = batch

        total_tokens = sum(
            [(nt + 1) / 2 for nt in num_transitions_batch.reshape(-1)])

        # Reset cached gradients.
        optimizer.zero_grad()

        if FLAGS.model_type in ["Pyramid", "ChoiPyramid"]:
            pyramid_temperature_multiplier = FLAGS.pyramid_temperature_decay_per_10k_steps ** (
                step / 10000.0)
            if FLAGS.pyramid_temperature_cycle_length > 0.0:
                min_temp = 1e-5
                pyramid_temperature_multiplier *= (
                    math.cos((step) / FLAGS.pyramid_temperature_cycle_length) + 1 + min_temp) / 2
        else:
            pyramid_temperature_multiplier = None

        # Run model.
        if step >= FLAGS.sbs_step: # gumbel-softmax
            dist_batch = None
        # else:
            # pass # sbs training

        output = model(
            X_batch,
            transitions_batch,
            y_batch,
            dist=dist_batch,
            use_internal_parser=FLAGS.use_internal_parser,
            validate_transitions=FLAGS.validate_transitions,
            pyramid_temperature_multiplier=pyramid_temperature_multiplier,
            example_lengths=num_transitions_batch)

        # Normalize output.
        logits = F.log_softmax(output)

        # Calculate class accuracy.
        target = torch.from_numpy(y_batch).long()

        # get the index of the max log-probability
        pred = logits.data.max(1, keepdim=False)[1].cpu()

        class_acc = pred.eq(target).sum() / float(target.size(0))

        # Calculate class loss.
        xent_loss = nn.NLLLoss()(logits, to_gpu(Variable(target, volatile=False)))

        # Optionally calculate transition loss.
        transition_loss = model.transition_loss if hasattr(
            model, 'transition_loss') else None

        # Extract L2 Cost
        l2_loss = get_l2_loss(
            model, FLAGS.l2_lambda) if FLAGS.use_l2_loss else None

        # Accumulate Total Loss Variable
        total_loss = 0.0
        total_loss += xent_loss
        if l2_loss is not None:
            total_loss += l2_loss
        if transition_loss is not None:
            total_loss += transition_loss

        #if FLAGS.sbs_loss:
        if hasattr(model, 'sbs_loss'):
            #print('yes, I have sbs_loss')
            total_loss +=  model.sbs_loss * FLAGS.sbs_weight

        aux_loss = auxiliary_loss(model)    # 0 aux_loss for gumbel tree

        total_loss += aux_loss



        # Backward pass.
        total_loss.backward()

        # Hard Gradient Clipping
        clip = FLAGS.clipping_max_value
        for p in model.parameters():
            if p.requires_grad:
                p.grad.data.clamp_(min=-clip, max=clip)

        # Learning Rate Decay
        if FLAGS.actively_decay_learning_rate:
            optimizer.lr = FLAGS.learning_rate * \
                (FLAGS.learning_rate_decay_per_10k_steps ** (step / 10000.0))

        # Gradient descent step.
        optimizer.step()

        end = time.time()

        total_time = end - start

        train_accumulate(model, data_manager, A, batch)
        A.add('class_acc', class_acc)
        A.add('total_tokens', total_tokens)
        A.add('total_time', total_time)

        if hasattr(model, 'sbs_loss'):
            A.add('sbs_cost', model.sbs_loss)

        if step % FLAGS.statistics_interval_steps == 0:
            A.add('xent_cost', xent_loss.data[0] )
            A.add('l2_cost', l2_loss.data[0])
            A.add('sbs_acc', model.sbs_acc.data[0])
            A.add('sbs_loss', model.sbs_loss.data[0])
            stats(model, optimizer, A, step, log_entry)
            should_log = True
            progress_bar.finish()


        if step % FLAGS.sample_interval_steps == 0 and FLAGS.num_samples > 0:
            should_log = True
            model.train()
            model(
                X_batch,
                transitions_batch,
                y_batch,
                use_internal_parser=FLAGS.use_internal_parser,
                validate_transitions=FLAGS.validate_transitions,
                pyramid_temperature_multiplier=pyramid_temperature_multiplier,
                example_lengths=num_transitions_batch)
            tr_transitions_per_example, tr_strength = model.spinn.get_transitions_per_example()

            model.eval()
            model(
                X_batch,
                transitions_batch,
                y_batch,
                use_internal_parser=FLAGS.use_internal_parser,
                validate_transitions=FLAGS.validate_transitions,
                pyramid_temperature_multiplier=pyramid_temperature_multiplier,
                example_lengths=num_transitions_batch)
            ev_transitions_per_example, ev_strength = model.spinn.get_transitions_per_example()

            if model.use_sentence_pair and len(transitions_batch.shape) == 3:
                transitions_batch = np.concatenate([
                    transitions_batch[:, :, 0], transitions_batch[:, :, 1]], axis=0)

            # This could be done prior to running the batch for a tiny speed
            # boost.
            t_idxs = range(FLAGS.num_samples)
            random.shuffle(t_idxs)
            t_idxs = sorted(t_idxs[:FLAGS.num_samples])
            for t_idx in t_idxs:
                log = log_entry.rl_sampling.add()
                gold = transitions_batch[t_idx]
                pred_tr = tr_transitions_per_example[t_idx]
                pred_ev = ev_transitions_per_example[t_idx]
                strength_tr = sparks(
                    [1] + tr_strength[t_idx].tolist(), dec_str)
                strength_ev = sparks(
                    [1] + ev_strength[t_idx].tolist(), dec_str)
                _, crossing = evalb.crossing(gold, pred_ev)
                log.t_idx = t_idx
                log.crossing = crossing
                log.gold_lb = "".join(map(str, gold))
                log.pred_tr = "".join(map(str, pred_tr))
                log.pred_ev = "".join(map(str, pred_ev))
                log.strg_tr = strength_tr[1:].encode('utf-8')
                log.strg_ev = strength_ev[1:].encode('utf-8')

        if step > 10000 and step % FLAGS.eval_interval_steps == 0:
            should_log = True
            for index, eval_set in enumerate(eval_iterators):
                acc, _, f1 = evaluate(
                    FLAGS, model, data_manager, eval_set, log_entry, logger, step, show_sample=(
                        step %
                        FLAGS.sample_interval_steps == 0), vocabulary=vocabulary, eval_index=index)
                if FLAGS.ckpt_on_best_dev_error and index == 0 and (
                        1 - acc) < 0.999 * best_dev_error and step > FLAGS.ckpt_step:
                    best_dev_error = 1 - acc
                    logger.Log(
                        "Checkpointing with new best dev accuracy of %f" %
                        acc)
                    trainer.save(best_checkpoint_path, step, best_dev_error, 1. - f1)
                    best_dev_step = step
                if index == 0 and (1 - f1) < 0.999 * best_dev_f1_error and step > FLAGS.ckpt_step:
                    best_dev_f1_error = 1 - f1
                    logger.Log(
                        "Checkpointing with new best dev f1 of %f" %
                        f1)
                    trainer.save(best_parsing_checkpoint_path, step, 1 - acc, best_dev_f1_error)
                if FLAGS.save_sl and step % FLAGS.save_sl_step == 0 and step < FLAGS.sbs_step:
                    logger.Log('save checkpoint')
                    trainer.save(get_checkpoint_path_for_sl(FLAGS.ckpt_path, FLAGS.experiment_name, step=step),
                        step, 1 - acc, 1. - f1)


            progress_bar.reset()

        if step > FLAGS.ckpt_step and step % FLAGS.ckpt_interval_steps == 0:
            should_log = True
            logger.Log("Checkpointing.")
            trainer.save(standard_checkpoint_path, step, best_dev_error, best_dev_f1_error)

        if should_log:
            logger.LogEntry(log_entry)

        progress_bar.step(i=(step % FLAGS.statistics_interval_steps) + 1,
                          total=FLAGS.statistics_interval_steps)


def run(only_forward=False):
    logger = afs_safe_logger.ProtoLogger(
        log_path(FLAGS), print_formatter=create_log_formatter(
            True, False), write_proto=FLAGS.write_proto_to_log)
    header = pb.SpinnHeader()


    data_manager = get_data_manager(FLAGS.data_type)

    logger.Log("Flag Values:\n" +
               json.dumps(FLAGS.FlagValuesDict(), indent=4, sort_keys=True))


    flags_dict = sorted(list(FLAGS.FlagValuesDict().items()))
    for k, v in flags_dict:
        flag = header.flags.add()
        flag.key = k
        flag.value = str(v)

    if not FLAGS.expanded_eval_only_mode:
        # Get Data and Embeddings for training
        preprocessed_data_path = os.path.join(FLAGS.ckpt_path, 
            'allnli_preprocessed_data_prpn-{}_train-{:d}-valid-{:d}_batch-{:d}_dist-{}.dat'.format(
                FLAGS.prpn_name, FLAGS.seq_length, FLAGS.eval_seq_length, FLAGS.batch_size, FLAGS.tree_joint))
        if os.path.isfile(preprocessed_data_path):
            print 'Reading dumped preprocessed data'
            vocabulary, initial_embeddings, picked_train_iter_pack, eval_iterators = cPickle.load(open(preprocessed_data_path, "rb" ))
        else:
            vocabulary, initial_embeddings, picked_train_iter_pack, eval_iterators = \
                load_data_and_embeddings(FLAGS, data_manager, logger,
                                        FLAGS.training_data_path, FLAGS.eval_data_path,
                                        )
            print 'Dumping data'
            cPickle.dump((vocabulary, initial_embeddings, picked_train_iter_pack, list(eval_iterators)), open(preprocessed_data_path, 'wb'))
            print 'Dumping done'
        train_sources, train_batches = picked_train_iter_pack

        def unpack_pickled_train_iter(sources, batches):
            '''
            '''
            num_batches = len(batches)
            idx = -1
            order = range(num_batches)
            random.shuffle(order)

            while True:
                idx += 1
                if idx >= num_batches:
                    # Start another epoch.
                    num_batches = len(batches)
                    idx = 0
                    order = range(num_batches)
                    random.shuffle(order)
                batch_indices = batches[order[idx]]
                # yield tuple(source[batch_indices] for source in sources if source is not None)
                yield tuple(source[batch_indices] if source is not None else None for source in sources)    # for gumbel tree model, the dist will be None
        training_data_iter = unpack_pickled_train_iter(train_sources, train_batches)

    else:
        # Get Data and Embeddings for test only
        vocabulary, initial_embeddings, training_data_iter, eval_iterators = \
            load_data_and_embeddings(FLAGS, data_manager, logger,
                                    FLAGS.training_data_path, FLAGS.eval_data_path,
                                    )

    # Build model.
    vocab_size = len(vocabulary)
    num_classes = len(set(data_manager.LABEL_MAP.values()))

    model, optimizer, trainer = init_model(
        FLAGS, logger, initial_embeddings, vocab_size, num_classes, data_manager, header)

    standard_checkpoint_path = get_checkpoint_path(
        FLAGS.ckpt_path, FLAGS.experiment_name)
    best_checkpoint_path = get_checkpoint_path(
        FLAGS.ckpt_path, FLAGS.experiment_name, best=True)
    best_parsing_checkpoint_path = get_checkpoint_path(
        FLAGS.ckpt_path, FLAGS.experiment_name, best=True, parsing=True)
    sl_checkpoint_path = get_checkpoint_path_for_sl(FLAGS.ckpt_path, FLAGS.experiment_name, step=FLAGS.load_sl_step)


    # Load checkpoint if available.
    if FLAGS.customize_ckpt:
        customize_ckpt_path = FLAGS.customize_ckpt_path
        logger.Log("Found pretrained customized checkpoint, restoring.")
        step, best_dev_error, best_dev_f1_error = trainer.load(customize_ckpt_path, cpu=FLAGS.gpu < 0, continue_train=FLAGS.continue_train)
        best_dev_step = 0
    elif FLAGS.load_best:
        if FLAGS.test_type == 'classification' and os.path.isfile(best_checkpoint_path):
            logger.Log("Found best classification checkpoint, restoring.")
            step, best_dev_error, dev_f1_error= trainer.load(best_checkpoint_path, cpu=FLAGS.gpu < 0)
            logger.Log(
                "Resuming at step: {} best dev accuracy: {} with dev f1: {}".format(
                    step, 1. - best_dev_error, 1. - dev_f1_error))
            step = 0
            best_dev_step = 0
            best_dev_f1_error = dev_f1_error
        elif os.path.isfile(best_parsing_checkpoint_path):
            logger.Log("Found best parsing checkpoint, restoring.")
            step, dev_error, best_dev_f1_error = trainer.load(best_parsing_checkpoint_path, cpu=FLAGS.gpu < 0)
            logger.Log(
                "Resuming at step: {} best f1: {} with dev accuracy: {}".format(
                    step, 1. - best_dev_f1_error, 1. - dev_error))
        else:
            raise ValueError('Can\'t find the best checkpoint.')
    elif FLAGS.load_sl:
        logger.Log("Found pretrained SL checkpoint at step {:d}, restoring.".format(FLAGS.load_sl_step))
        step, best_dev_error, best_dev_f1_error = trainer.load(standard_checkpoint_path, cpu=FLAGS.gpu < 0, continue_train=FLAGS.continue_train)
        best_dev_step = 0
    elif os.path.isfile(standard_checkpoint_path):
        logger.Log("Found checkpoint, restoring.")
        step, best_dev_error, best_dev_f1_error = trainer.load(standard_checkpoint_path, cpu=FLAGS.gpu < 0)
        logger.Log(
            "Resuming at step: {} previously best dev accuracy: {} and previously best f1: {}".format(
                step, 1. - best_dev_error, 1. - best_dev_f1_error))
    else:
        assert not only_forward, "Can't run an eval-only run without a checkpoint. Supply a checkpoint."
        step = 0
        best_dev_error = 1.0
        best_dev_step = 0
        best_dev_f1_error = 1.0    # for best parsing checkpoint
    header.start_step = step
    header.start_time = int(time.time())

    # # Right-branching trick.
    # DefaultUniformInitializer(model.binary_tree_lstm.comp_query.weight)
    # set temperature
    model.binary_tree_lstm.temperature_param.data = torch.Tensor([[0.2]])

    # GPU support.
    the_gpu.gpu = FLAGS.gpu
    if FLAGS.gpu >= 0:
        model.cuda()
    else:
        model.cpu()
    recursively_set_device(optimizer.state_dict(), FLAGS.gpu)

    # Debug
    def set_debug(self):
        self.debug = FLAGS.debug
    model.apply(set_debug)

    # Do an evaluation-only run.
    logger.LogHeader(header)  # Start log_entry logging.
    if only_forward:
        log_entry = pb.SpinnEntry()
        for index, eval_set in enumerate(eval_iterators):
            log_entry.Clear()
            evaluate(
                FLAGS,
                model,
                data_manager,
                eval_set,
                log_entry,
                logger,
                step,
                vocabulary,
                show_sample=True,
                eval_index=index)
            print(log_entry)
            logger.LogEntry(log_entry)
    else:
        best_dev_step = 0
        train_loop(
            FLAGS,
            data_manager,
            model,
            optimizer,
            trainer,
            training_data_iter,
            eval_iterators,
            logger,
            step,
            best_dev_error,
            best_dev_step,
            best_dev_f1_error,
            vocabulary)


if __name__ == '__main__':
    get_flags()

    # Parse command line flags.
    FLAGS(sys.argv)

    flag_defaults(FLAGS)

    print 'running...'
    if FLAGS.model_type == "RLSPINN":
        raise Exception(
            "Please use rl_classifier.py instead of supervised_classifier.py for RLSPINN.")

    run(only_forward=FLAGS.expanded_eval_only_mode)
