# PCL Python API Guide

Python interface for configuring, training, and using PCL networks.

---

## 📋 Table of Contents

1. [Installation](#installation)
2. [ConfigBuilder API](#configbuilder-api)
   - [Basic Setup](#basic-setup)
   - [Dedicated Configuration Methods](#dedicated-configuration-methods)
   - [Generic Parameter Configuration](#generic-parameter-configuration)
   - [Complete Configuration Example](#complete-configuration-example)
3. [Network Operations](#network-operations)
   - [Creating a Network](#creating-a-network)
   - [Training and Inference](#training-and-inference)
4. [Spike Recording](#spike-recording)
   - [Dataset Organization](#dataset-organization)
   - [Configuration Example](#configuration-example)
   - [Running Spike Recording](#running-spike-recording)
   - [Output Structure](#output-structure)
   - [Loading Recorded Data](#loading-recorded-data)
5. [Visualization](#visualization)

---

## 🛠️ Installation

```bash
# Create virtual environment
python -m venv PCL_env
source PCL_env/bin/activate  # On Windows: PCL_env\Scripts\activate

# Install dependencies (if not already done)
# ...

# Build C++ executable
cd ../build
cmake .. && make
```

---

## ⚙️ ConfigBuilder API

Python interface for network configuration with fluent/chainable syntax.

### Basic Setup

```python
from src.spiking_network.planning import ConfigBuilder

# Load existing network configuration
config = ConfigBuilder.load("/path/to/network/configs")

# Create new configuration with default values
config = ConfigBuilder.default()

# Create from dictionary
config_dict = {...}
config = ConfigBuilder.from_dict(config_dict)
```

### Dedicated Configuration Methods

The ConfigBuilder provides specialized setter methods for common network parameters.

#### Network Architecture

```python
config = (ConfigBuilder.load("/path/to/network/configs")
          .set_visual_field(66, 66)                    # Input dimensions
          .set_layer_sizes([[9, 9, 64], [6, 6, 32]])   # [width, height, depth] per layer
          .set_neuron_sizes([[[10, 10, 1]], [[4, 4, 64]]])  # Receptive field sizes
          .set_neuron_overlap([[3, 3, 0], [3, 3, 0]])  # Overlap between neurons
          .set_layer_patches([[[0], [0], [0]], [[0], [0], [0]]])  # Patch configurations
          .save())
```

#### Spike Recording

```python
# Enable and configure spike recording
config = (ConfigBuilder.load(network_path)
          .enable_spike_recording()                    # Enable recording
          .record_layers([0, 1])                       # Specific layers (empty = all)
          .set_max_samples_per_class(1000)             # 0 = unlimited
          .set_output_subfolder("experiment_01")       # Output folder name
          .save())

# Disable spike recording
config = config.disable_spike_recording().save()
```

### Generic Parameter Configuration

For parameters without dedicated setter methods, use `.with_param(config_name, param_name, value)`:

- **config_name**: Target configuration file (`"network"`, `"simple_cell"`, `"complex_cell"`)
- **param_name**: Parameter name as defined in JSON config
- **value**: New value for the parameter

#### Learning Parameters (STDP)

```python
# Simple cell learning parameters
config = (ConfigBuilder.load(network_path)
          .with_param("simple_cell", "ETA_LTP", 0.0003)      # LTP learning rate
          .with_param("simple_cell", "ETA_LTD", -0.0003)     # LTD learning rate
          .with_param("simple_cell", "STDP_LEARNING", "both") # Learning mode
          .with_param("simple_cell", "TAU_M", 20000)         # Membrane time constant
          .with_param("simple_cell", "VTHRESH", 10.0)        # Spike threshold
          .save())

# Complex cell learning parameters
config = (config
          .with_param("complex_cell", "ETA_LTP", 0.0003)
          .with_param("complex_cell", "STDP_LEARNING", "both")
          .with_param("complex_cell", "SIGMA", 5000)         # Temporal window
          .save())
```

#### Inhibition Configuration

```python
# Full PCL with all inhibition types (local, lateral, top-down)
config = (ConfigBuilder.load(network_path)
          .with_param("network", "layerInhibitions", 
                     [["local", "lateral"],                  # Layer 0 inhibitions
                      ["local", "lateral", "topdown"]])      # Layer 1 inhibitions
          .with_param("network", "neuronInhibitionRange", [4, 4])  # Inhibition radius
          .save())

# Hierarchical without top-down
config = (config
          .with_param("network", "layerInhibitions",
                     [["local", "lateral"],
                      ["local", "lateral"]])
          .save())
```

### Complete Configuration Example

```python
from src.spiking_network.planning import ConfigBuilder

# Comprehensive network configuration
config = (ConfigBuilder.load("../networks/my_network/configs")
          # Architecture (dedicated methods)
          .set_visual_field(66, 66)
          .set_layer_sizes([[9, 9, 64], [6, 6, 32]])
          .set_neuron_sizes([[[10, 10, 1]], [[4, 4, 64]]])
          .set_neuron_overlap([[3, 3, 0], [3, 3, 0]])
          # Inhibition (generic with_param)
          .with_param("network", "layerInhibitions",
                     [["local", "lateral"],
                      ["local", "lateral", "topdown"]])
          .with_param("network", "neuronInhibitionRange", [4, 4])
          # Learning (generic with_param)
          .with_param("simple_cell", "ETA_LTP", 0.0003)
          .with_param("simple_cell", "STDP_LEARNING", "both")
          .with_param("complex_cell", "ETA_LTP", 0.0003)
          .with_param("complex_cell", "STDP_LEARNING", "both")
          # Spike recording (dedicated methods)
          .enable_spike_recording()
          .set_output_subfolder("experiment_01")
          # Save all changes
          .save())
```

---

## 🔧 Network Operations

### Creating a Network

```python
from src.spiking_network.planning import create_network

# Create new network with default configuration
create_network(
    exec_path="/path/to/neuvisys-exe",
    network_path="/path/to/network/configs"
)
```

### Training and Inference

```python
from src.spiking_network.planning import launch_neuvisys_multi_pass

# Train network on event dataset (5 epochs/passes)
launch_neuvisys_multi_pass(
    exec_path="../../build/libs/apps/neuvisys/neuvisys-exe",
    network_path="../networks/my_network/configs",
    event_file="../../data/events/dataset.h5",  # Single file for training
    nb_pass=5
)

# For spike recording, use dataset directory instead (see Spike Recording section)
```

---

## 📊 Spike Recording

Record spike activity for analysis and classification tasks.

### Dataset Organization

Organize your dataset in class-labeled folders for automatic label assignment:

```
dataset/
├── 0_class_name/         # Class label 0 (numeric prefix optional)
│   ├── sample001.h5
│   ├── sample002.npz
│   └── ...
├── 1_another_class/      # Class label 1
│   ├── sample001.h5
│   └── ...
└── 2_third_class/        # Class label 2
    └── ...
```

**Supported formats:** `.h5`, `.npz`

### Configuration Example

```python
from src.spiking_network.planning import ConfigBuilder

# Enable spike recording with custom settings
config = (ConfigBuilder.load("/path/to/network/configs")
          .enable_spike_recording()
          .record_layers([0, 1])                        # Specific layers (empty = all)
          .set_max_samples_per_class(1000)              # Limit samples (0 = unlimited)
          .set_output_subfolder("gesture_spikes")       # Output folder name
          .save())
```

### Running Spike Recording

```python
from src.spiking_network.planning import launch_neuvisys_multi_pass

# Record spikes for dataset
launch_neuvisys_multi_pass(
    exec_path="../../build/libs/apps/neuvisys/neuvisys-exe",
    network_path="../networks/my_network/configs",
    event_file="../../data/events/gesture_dataset",   # Directory with class folders
    nb_pass=1                                          # Typically 1 pass for recording
)
```

### Output Structure

Recorded spike data is saved as aggregated descriptors in a simple hierarchical structure:

```
network_path/statistics/
└── <outputSubfolder>/           # e.g., "gesture", "mnist"
    ├── 0/                        # Class label 0
    │   ├── 0.json                # Sample 0
    │   ├── 1.json                # Sample 1
    │   ├── 2.json                # Sample 2
    │   └── ...
    ├── 1/                        # Class label 1
    │   ├── 0.json
    │   ├── 1.json
    │   └── ...
    ├── 2/                        # Class label 2
    │   └── ...
    └── ...
```

**Example:** `networks/my_network/statistics/gesture/0/5.json` contains spike descriptor data for class 0, sample 5.

**JSON structure:**
Each JSON file contains spike descriptors with:
- `x`: Feature vectors (spike counts/patterns across neurons and layers)
- `y`: Label information

### Loading Recorded Data

Todo

---

## 📈 Visualization

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