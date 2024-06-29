from pathlib import Path
import moftransformer
from moftransformer.examples import example_path

root_dataset = example_path['root_dataset']
downstream = example_path['downstream']

# Get ckpt file
seed = 0               # default seeds
version = 0            # version for model. It increases with the number of trains

# For version > 2.1.1, best.ckpt exists
checkpoint = 'best'    # Epochs where the model is stored. 
log_dir = './logs/'
save_dir = 'result/'

# optional keyword
mean = 0
std = 1

load_path = Path(log_dir) / f'pretrained_mof_seed{seed}_from_pmtransformer/version_{version}/checkpoints/{checkpoint}.ckpt'

if not load_path.exists():
    raise ValueError(f'load_path does not exists. check path for .ckpt file : {load_path}')

moftransformer.test(root_dataset, load_path, downstream=downstream,
                   save_dir=save_dir, mean=mean, std=std)