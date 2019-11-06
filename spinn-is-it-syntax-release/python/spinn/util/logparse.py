"""

Really easy log parsing.

"""

try:
    from parse import *
except BaseException:
    pass
import json
import sys

def parse_float_number(text):
    return float(text)

FMT_TRAIN = "Step: {step:Num} Acc: classfify {cl_acc:Num} tr(invalid) {} sbs {}" + \
    " Loss: total {} xent {} tr {} reg {} sbs {} Time: {}"
FMT_EVAL = "Step: {step:Num} Eval acc: cl {cl_acc:Num} f1: {f1:Num} tr {} {} Time: {}"

IS_TRAIN = "Acc:"
IS_TRAIN_EXTRA = "Train Extra:"
IS_EVAL = "Eval acc:"
IS_EVAL_EXTRA = "Eval Extra:"

START_TRAIN = "Step:"
START_TRAIN_EXTRA = "Train Extra:"
START_EVAL = "Step:"
START_EVAL_EXTRA = "Eval Extra:"


def get_format(filename, prefix):
    with open(filename) as f:
        for line in f:
            if prefix in line:
                return line[line.find(prefix) + len(prefix):].strip()
    raise Exception("Format string not found.")


def parse_flags(filename):
    PREFIX_FLAGS = "Flag Values:\n"
    TERMINAL = "}\n"
    data = ""
    read_json = False
    with open(filename) as f:
        for line in f:
            if read_json:
                data += line
                if TERMINAL in line:
                    break
            if PREFIX_FLAGS in line:
                read_json = True
    return json.loads(data)


def is_train(line):
    return line.find(IS_TRAIN) >= 0


def is_eval(line):
    return line.find(IS_EVAL) >= 0


def read_file(filename):
    flags = parse_flags(filename)
    # train_str = get_format(filename, FMT_TRAIN)
    # eval_str = get_format(filename, FMT_EVAL)

    dtrain, deval = [], []

    with open(filename) as f:
        for line in f:
            line = line.strip()
            if is_train(line):
                dtrain.append(parse(FMT_TRAIN,
                                    line[line.find(START_TRAIN):].strip(), dict(Num=parse_float_number)))
            elif is_eval(line):
                deval.append(parse(FMT_EVAL,
                                   line[line.find(START_EVAL):].strip(), dict(Num=parse_float_number)))
            else:
                pass

    return dtrain, deval, flags


if __name__ == '__main__':
    import gflags
    import sys

    FLAGS = gflags.FLAGS
    gflags.DEFINE_string("log_path", "", "")
    gflags.DEFINE_string("csv_path", "", "")
    FLAGS(sys.argv)

    dtrain, deval, flags = read_file(FLAGS.log_path)

    print "Flags:"
    print "Model={model_type}\nLearning_Rate={learning_rate}".format(**flags)
    print

    step = []
    eval_cl = []
    eval_f1_cl = []

    print "Train:"
    for d in dtrain:
        print("Step: {:d} Acc: {:.5f}".format(int(d['step']), d['cl_acc']))
    print

    print "Eval:"
    for d in deval:
        cur_step = int(d['step'])
        cur_eval_cl_acc = d['cl_acc']
        cur_eval_f1 = d['f1']
        step.append(cur_step)
        eval_cl.append(cur_eval_cl_acc)
        eval_f1_cl.append(cur_eval_f1)
        print("Step: {:d} Acc: {:.5f} F1: {:.5f}".format(cur_step, cur_eval_cl_acc, cur_eval_f1))

    with open(FLAGS.csv_path, 'w') as fw:
        for x, y, z in zip(step, eval_cl, eval_f1_cl):
            fw.write('{:d}, {:.5f}, {:.5f}\n'.format(x, y, z))
