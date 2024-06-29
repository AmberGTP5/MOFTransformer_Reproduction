import moftransformer
from moftransformer.examples import example_path

# data root and downstream from example
root_dataset = example_path['root_dataset']
downstream = example_path['downstream']
log_dir = './logs/'
# load_path = "pmtransformer" (default)

# kwargs (optional)
max_epochs = 10
batch_size = 8
mean = 0
std = 1


moftransformer.run(root_dataset, downstream, log_dir=log_dir,                   
                   max_epochs=max_epochs, batch_size=batch_size,
                   mean=mean, std=std)