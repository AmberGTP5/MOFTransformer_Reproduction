# MOFTransformer version 2.1.1
from itertools import chain
from moftransformer.run import run

str_kwargs_names = {
    'load_path': 'This parameter specifies the path of the model that will be used for training/testing. '
    'The available options are "pmtransformer", "moftransformer", other .ckpt paths, and None (scratch). '
    "If you want to test a fine-tuned model, you should specify the path to the .ckpt file stored in the 'log' folder."
    "To download a pre-trained model, use the following command:"
    "$ moftransformer download pretrain_model'",
    'loss_names': "One or more of the following loss : 'regression', 'classification', 'mpt', 'moc', and 'vfp'",
    'optim_type': 'Type of optimizer, which is "adamw", "adam", or "sgd" (momentum=0.9)',
    'resume_from' : 'Restart model - path for ckpt'
}

int_kwargs_names = {
    'n_classes': "Number of classes when your loss is 'classification'",
    'batch_size': 'desired batch size; for gradient accumulation',
    'per_gpu_batchsize': 'you should define this manually with per_gpu_batch_size',
    'num_nodes': 'Number of GPU nodes for distributed training.',
    'num_workers': "the number of cpu's core",
    'max_epochs': 'Stop training once this number of epochs is reached.',
    'seed': 'The random seed for pytorch_lightning.',
    'max_steps': 'num_data * max_epoch // batch_size (accumulate_grad_batches). If -1, set max_steps automatically.',
}

float_kwargs_names = {
    'mean': 'mean for normalizer. If None, it is automatically obtained from the train dataset.',
    'std': 'standard deviation for normalizer. If None, it is automatically obtained from the train dataset.',
    'learning_rate': 'Learning rate for optimizer',
    'weight_decay': 'Weight decay for optmizer',
    'decay_power': 'default polynomial decay, [cosine, constant, constant_with_warmup]',
    'warmup_steps' : 'warmup steps for optimizer. If type is float, set to max_steps * warmup_steps.',
}

bool_kwargs_names = {
    'visualize': 'return attention map (use at attetion visualization step)',
}

none_kwargs_names = {
    'accelerator' : 'Supports passing different accelerator types ("cpu", "gpu", "tpu", "ipu", "hpu", "mps, "auto") '
    'as well as custom accelerator instances.',
    'devices' : 'Number of devices to train on (int), which devices to train on (list or str), or "auto". '
    'It will be mapped to either gpus, tpu_cores, num_processes or ipus, based on the accelerator type ("cpu", "gpu", "tpu", "ipu", "auto").',
}


class CLICommand:
    """
    run moftransformer code

    ex) moftransformer run downstream=example downstream=bandgap devices=1 max_epochs=10

    """

    @staticmethod
    def add_arguments(parser):
        #parser.add_argument("args", nargs="*")
        parser.add_argument("--root_dataset", "-r", type=str, 
                            help="A folder containing graph data, grid data, and json of MOFs that you want to train or test. "\
                            "The way to make root_dataset is at this link (https://hspark1212.github.io/MOFTransformer/dataset.html)"
                            )
        parser.add_argument('--downstream', "-d", type=str, default=None, help="Name of user-specific task (e.g. bandgap, gasuptake, etc). "
                            "if downstream is None, target json is 'train.json', 'val.json', and 'test.json'",
                            )
        parser.add_argument('--log_dir', "-l", default='./logs', type=str,
                            help='(optional) Directory to save log, models, and params. (default = ./logs/)'
                            )
        parser.add_argument('--test_only', "-t", action='store_true',
                            help='(optional) If True, only the test process is performed without the learning model.'
                            )

        for key, value in str_kwargs_names.items():
            parser.add_argument(f"--{key}", type=str, required=False, help=f"(optional) {value}")

        for key, value in none_kwargs_names.items():
            parser.add_argument(f"--{key}", required=False, help=f"(optional) {value}")

        for key, value in int_kwargs_names.items():
            parser.add_argument(f"--{key}", type=int, required=False, help=f"(optional) {value}")

        for key, value in float_kwargs_names.items():
            parser.add_argument(f"--{key}", type=float, required=False, help=f"(optional) {value}")

        for key, value in bool_kwargs_names.items():
            parser.add_argument(f"--{key}", action='store_true', required=False, help=f"(optional) {value}")

    @staticmethod
    def run(args):
        from moftransformer import __root_dir__
        
        root_dataset = args.root_dataset
        downstream = args.downstream
        log_dir = args.log_dir
        test_only = args.test_only

        kwargs = {}
        for key in chain(str_kwargs_names.keys(), 
                         none_kwargs_names.keys(),
                         int_kwargs_names.keys(),
                         float_kwargs_names.keys(),
                         bool_kwargs_names.keys(),
                         ):
            if value := getattr(args, key):
                kwargs[key] = value
                
        run(
            root_dataset,
            downstream,
            log_dir,
            test_only=test_only,
            **kwargs
        )
