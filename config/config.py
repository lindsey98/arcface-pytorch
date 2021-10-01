
class Config(object):

    env = 'default'
    backbone = 'resnet50'
    classify = 'softmax'
    metric = 'arc_margin'
    easy_margin = False
    use_se = False
    loss = 'focal_loss'

    display = False
    finetune = False

    dataset = 'logo2k_super100'
    num_classes = 107
    train_root = '/home/ruofan/PycharmProjects/ProxyNCA-/mnt/datasets/logo2ksuperclass0.01'
    test_root = '/home/ruofan/PycharmProjects/ProxyNCA-/mnt/datasets/logo2ksuperclass0.01'
    checkpoints_path = 'checkpoints'
    load_model_path = 'models/resnet50_logo2k_super100.pth'
    test_model_path = 'checkpoints/resnet50_logo2k_super100.pth'
    save_interval = 10
    nb_workers = 8

    sz_batch = 32  # batch size
    test_batch_size = 64
    IPC = 8

    input_shape = (3, 224, 224)

    optimizer = 'sgd'

    use_gpu = True  # use GPU or not
    gpu_id = '1'
    num_workers = 0  # how many workers for loading data
    print_freq = 100  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 50
    lr = 1e-4  # initial learning rate
    lr_step = 10
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 5e-4
