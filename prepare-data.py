from examples import example_path
from utils import prepare_data

# Get example path
root_cifs = example_path['root_cif']
root_dataset = example_path['root_dataset']
downstream = example_path['downstream']

train_fraction = 0.8  # default value
test_fraction = 0.1   # default value

# Run prepare data
prepare_data(root_cifs, root_dataset, downstream=downstream, 
             train_fraction=train_fraction, test_fraction=test_fraction)