class DefaultConfig(object):
    VOC_SIZE = 695
    alpha = 0.1
    max_length = 137
    input_size = 695
    hidden_size = 128
    output_size = 695
    seq_size = 46
    DA_size = 162
    env = 'slot-generation'
    epoch = 100
    lr = 5.0
    alpha = 0.1
    batch_size = 100
    best_acc = 0
    gen_time = 30
    START_ix = 0
    END_ix = 1
    AND_ix = VOC_SIZE  # means add AND_ix notion as extra data
    eta = 0.0001
    kexi = 100
    lamda = 0.1



opt = DefaultConfig()
if __name__ == '__main__':
    print(opt.VOC_SIZE)