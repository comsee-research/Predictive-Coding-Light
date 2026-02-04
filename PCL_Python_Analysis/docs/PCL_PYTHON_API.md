# PCL Python API Guide

Python interface for configuring, training, and using PCL networks.

## Installation

```bash
# Create virtual environment
python -m venv PCL_env
source PCL_env/bin/activate  # On Windows: PCL_env\Scripts\activate

# Install dependencies

# Build C++ executable
cd ../build
cmake .. && make
```

---

## ConfigBuilder API

Python interface for network configuration with fluent/chainable syntax.

### Import

```python
from src.spiking_network.planning import ConfigBuilder
```

### Loading and Creating network configuration

```python
# Load existing network
config = ConfigBuilder.load("/path/to/network")

# Create with defaults
config = ConfigBuilder.default()

# Create from dictionary
config_dict = {...}
config = ConfigBuilder.from_dict(config_dict)
```

### Network Architecture

```python
config = (ConfigBuilder.load("/path/to/network")
          .set_visual_field(66, 66)
          .set_layer_sizes([[9, 9, 64], [6, 6, 32]])
          .set_neuron_sizes([[[10, 10, 1]], [[4, 4, 64]]])
          .set_neuron_overlap([[3, 3, 0], [3, 3, 0]])
          .set_layer_patches([[[0], [0], [0]], [[0], [0], [0]]])
          .save())
```

### Learning Configuration

```python
# Simple cell parameters
config = (ConfigBuilder.load(network_path)
          .with_param("simple_cell", "ETA_LTP", 0.0003)
          .with_param("simple_cell", "ETA_LTD", -0.0003)
          .with_param("simple_cell", "STDP_LEARNING", "both")
          .with_param("simple_cell", "TAU_M", 20000)
          .with_param("simple_cell", "VTHRESH", 10.0)
          .save())

# Complex cell parameters
config = (config
          .with_param("complex_cell", "ETA_LTP", 0.0003)
          .with_param("complex_cell", "STDP_LEARNING", "both")
          .with_param("complex_cell", "SIGMA", 5000)
          .save())
```

### Inhibition Configuration

```python
# Full PCL with all inhibition types
config = (ConfigBuilder.load(network_path)
          .with_param("network", "layerInhibitions", 
                     [["local", "lateral"], 
                      ["local", "lateral", "topdown"]])
          .with_param("network", "neuronInhibitionRange", [4, 4])
          .save())

# Simple hierarchical (no top-down)
config = (config
          .with_param("network", "layerInhibitions",
                     [["local", "lateral"],
                      ["local", "lateral"]])
          .save())
```

### Spike Recording Configuration

```python
# Enable spike recording
config = (ConfigBuilder.load(network_path)
          .enable_spike_recording()
          .record_layers([0, 1])  # Empty list = all layers
          .set_max_samples_per_class(1000)  # 0 = unlimited
          .set_output_subfolder("experiment_01")
          .save())

# Disable spike recording
config = config.disable_spike_recording().save()

# OR ...
# Quick functions
from src.spiking_network.planning import quick_enable_spike_recording

quick_enable_spike_recording(
    network_path="/path/to/network",
    layers=[0, 1],
    output_folder="spikes"
)
```

### Complete Example

```python
from src.spiking_network.planning import ConfigBuilder

# Create and configure network
config = (ConfigBuilder.load("../networks/my_network")
          # Architecture
          .set_visual_field(66, 66)
          .set_layer_sizes([[9, 9, 64], [6, 6, 32]])
          .set_neuron_sizes([[[10, 10, 1]], [[4, 4, 64]]])
          .set_neuron_overlap([[3, 3, 0], [3, 3, 0]])
          # Inhibition
          .with_param("network", "layerInhibitions",
                     [["local", "lateral"],
                      ["local", "lateral", "topdown"]])
          .with_param("network", "neuronInhibitionRange", [4, 4])
          # Learning
          .with_param("simple_cell", "ETA_LTP", 0.0003)
          .with_param("simple_cell", "STDP_LEARNING", "both")
          .with_param("complex_cell", "ETA_LTP", 0.0003)
          .with_param("complex_cell", "STDP_LEARNING", "both")
          # Save
          .save())
```

---

## Training & Execution

### Network Creation

```python
from src.spiking_network.planning import create_network

# Create new network with default configuration
create_network(
    exec_path="/path/to/neuvisys-exe",
    network_path="/path/to/network"
)
```

### Training/Execution

```python
from src.spiking_network.planning import launch_neuvisys_multi_pass

# Train on event dataset
launch_neuvisys_multi_pass(
    exec_path="../../build/libs/apps/neuvisys/neuvisys-exe",
    network_path="../networks/my_network",
    event_file="../../data/events/dataset.h5",  # or directory path
    nb_pass=1
)
```

### Command-Line Execution

```bash
# Training
./neuvisys-exe <network_path> <event_file> <nb_pass>

# Spike recording (requires config)
./neuvisys-exe <network_path> <dataset_path> 1
```

---

## Spike Recording

### Dataset Organization

Organize data in class folders for automatic label detection:

```
dataset/
├── 0_class_name/     # Numeric prefix (optional but recommended)
│   ├── sample001.h5
│   ├── sample002.npz
│   └── ...
├── 1_another_class/
│   └── ...
└── 2_third_class/
    └── ...
```

Supported formats: `.h5`, `.npz`

### Configuration

```python
from src.spiking_network.planning import ConfigBuilder

config = (ConfigBuilder.load("/path/to/network")
          .enable_spike_recording()
          .record_layers([0, 1])
          .set_output_subfolder("gesture_spikes")
          .save())
```

### Execution

```python
launch_neuvisys_multi_pass(
    exec_path="../../build/libs/apps/neuvisys/neuvisys-exe",
    network_path="../networks/my_network",
    event_file="../../data/events/gesture_dataset",  # Directory with class folders
    nb_pass=1
)
```

### Output Structure

```
network_path/statistics/<outputSubfolder>/
├── 0/              # Class 0
│   ├── 0/          # Sample 0
│   │   └── 1/
│   │       ├── 0.json    # Layer 0, Neuron 0
│   │       ├── 1.json    # Layer 0, Neuron 1
│   │       └── ...
│   ├── 1/          # Sample 1
│   └── ...
├── 1/              # Class 1
└── ...
```

### Loading Recorded Data
Todo

### Visualization

```python
from src.spiking_network.utils.visualization_utils import (
    visualize_receptive_fields,
    visualize_lateral_inhibition,
    visualize_topdown_connections
)

# Visualize receptive fields
visualize_receptive_fields(network, layer_id=0, save_dir="./figures/")

# Visualize lateral inhibition
visualize_lateral_inhibition(network, layer_id=0)

# Visualize top-down connections
visualize_topdown_connections(network, layer_id=0)
```
---

## API Reference Summary

### ConfigBuilder Methods

**Creation:**
- `ConfigBuilder.load(path)` - Load existing network
- `ConfigBuilder.default()` - Create with defaults
- `ConfigBuilder.from_dict(config_dict)` - Create from dictionary

**Network Architecture:**
- `.set_visual_field(width, height)`
- `.set_layer_sizes(sizes)`
- `.set_neuron_sizes(sizes)`
- `.set_neuron_overlap(overlap)`
- `.set_layer_patches(patches)`

**Generic Configuration:**
- `.with_param(config_type, param_name, value)`
  - `config_type`: `"network"`, `"simple_cell"`, or `"complex_cell"`

**Spike Recording:**
- `.enable_spike_recording()`
- `.disable_spike_recording()`
- `.record_layers(layers)`
- `.record_all_layers()`
- `.set_max_samples_per_class(max_samples)`
- `.set_output_subfolder(folder_name)`

**Query:**
- `.is_spike_recording_enabled()` → bool
- `.get_layers_to_record()` → list

**Persistence:**
- `.save()` - Save to disk
- `.build()` - Get config dictionaries

### Training Functions

- `create_network(exec_path, network_path)` - Create new network
- `launch_neuvisys_multi_pass(exec_path, network_path, event_file, nb_pass)` - Train/infer
- `quick_enable_spike_recording(path, layers, output_folder)` - Quick spike config