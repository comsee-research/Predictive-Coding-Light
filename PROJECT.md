# Predictive Coding Light (PCL) Project

## Overview

The **Predictive Coding Light (PCL)** project implements a biologically-inspired Spiking Neural Network (SNN) based on predictive coding principles. It consists of a high-performance C++ core library (Neuvisys) with a comprehensive Python wrapper for training, analysis, and visualization.

## Core Architecture

### C++ Core (Neuvisys Library)

**Location**: `libs/network/`

The core implements a hierarchical spiking neural network with two neuron types:

- **SimpleNeuron**: First-layer neurons that learn input features via STDP (Spike-Timing-Dependent Plasticity)
- **ComplexNeuron**: Higher-layer neurons that integrate simple cell responses and implement predictive feedback

**Key Classes**:
- `SpikingNetwork` - Main network structure managing layers and connectivity
- `NetworkHandle` - High-level API for event processing and training
- `Neuron` - Base class with membrane dynamics and synaptic plasticity

**Build System**: CMake with GoogleTest for testing

### Python Analysis Tools

**Location**: `PCL_Python_Analysis/`

Provides a complete workflow for:
- Event data preprocessing (`new_src/datasets.py`)
- Network configuration and training (`src/spiking_network/planning/`)
- Weight visualization and analysis (`src/spiking_network/analysis/`)
- Receptive field characterization and Gabor fitting

## Network Architecture

### Three Inhibition Types

The PCL network implements three complementary inhibition mechanisms:

1. **Local Inhibition** (`"local"`): Winner-take-all competition between neurons at the same spatial location
2. **Lateral Inhibition** (`"lateral"`): Spatial competition within a defined neighborhood range
3. **Top-Down Inhibition** (`"topdown"`): Predictive feedback from higher to lower layers

### Layer Structure

Networks consist of hierarchical layers with:
- Configurable receptive field sizes and overlaps
- Shared weight matrices for translation invariance
- STDP-based unsupervised learning
- Dynamic excitatory/inhibitory balance

## Data Flow

```
Event Input (HDF5/NPZ) 
  → Preprocessing (scaling, centering)
  → SpikingNetwork (C++ core)
  → STDP Learning + Inhibition
  → Weight Updates
  → Python Analysis/Visualization
```

## Key Features

### C++ Core
- Event-driven spike processing with priority queues
- Leaky integrate-and-fire neuron dynamics
- Multiple STDP learning rules (LTP/LTD)
- Membrane potential decay and refractoriness
- Efficient weight sharing and normalization

### Python Wrapper
- Network creation and configuration via JSON
- Multi-pass training on event datasets
- Visualization of learned receptive fields
- Statistical analysis (spike trains, tuning curves)
- Gabor filter fitting for orientation selectivity

## Configuration

Networks are configured through JSON files:

- `network_config.json` - Layer sizes, connectivity, visual field dimensions
- `simple_cell_config.json` - Simple neuron parameters (STDP rates, thresholds)
- `complex_cell_config.json` - Complex neuron parameters

Critical parameters:
- `neuronSizes` - Receptive field dimensions per layer
- `layerSizes` - Number of neurons per layer (width × height × features)
- `neuronOverlap` - RF overlap for spatial continuity
- `layerInhibitions` - Inhibition types per layer
- `ETA_LTP/ETA_LTD` - Learning rates for weight updates

## Datasets

Supported event-based datasets:
- **DVS-Gesture128**: Dynamic gesture recognition
- **Natural Images**: Static natural image patches converted to events

Event format: HDF5 files with groups containing timestamps (t), coordinates (x, y), polarities (p), and camera IDs (c)

## Usage Example

```python
from src.spiking_network.planning.network_planner import create_network, launch_neuvisys_multi_pass

# Create network structure
create_network(exec_path="/path/to/neuvisys-exe", network_path="./my_network")

# Configure network (modify JSON files)
# ...

# Train on event data
launch_neuvisys_multi_pass(
    exec_path="/path/to/neuvisys-exe",
    network_path="./my_network",
    event_file="data.h5",
    nb_pass=5
)

# Analyze results
from src.spiking_network.network.neuvisys import SpikingNetwork
network = SpikingNetwork("./my_network/")
network.visualize_weights(layer=0)
```

## Project Structure

```
PCL/
├── libs/network/          # C++ core implementation
│   ├── include/          # Header files (neurons, network classes)
│   └── src/              # Implementation (STDP, dynamics, connections)
├── PCL_Python_Analysis/   # Python tools
│   ├── src/              # Core Python packages
│   │   ├── events/       # Event file I/O and conversion
│   │   └── spiking_network/  # Network interface and analysis
│   ├── new_src/          # Dataset preprocessing utilities
│   ├── networks/         # Saved trained networks
│   └── *.ipynb           # Analysis notebooks
├── data/events/          # Event datasets (HDF5/NPZ)
├── externals/            # Dependencies (cnpy, json, libserie)
├── build/                # CMake build directory
└── tools/                # Installation scripts
```

## Dependencies

**C++**:
- OpenCV (image processing)
- HDF5 (event file I/O)
- Eigen (linear algebra, via dependencies)
- Qt5 + QtCharts (optional GUI)

**Python**:
- numpy, scipy (numerical computing)
- h5py (HDF5 file handling)
- matplotlib (visualization)
- scikit-learn (SVM classification)
- tqdm (progress bars)

## Scientific Basis

The PCL network models visual cortex mechanisms:
- **Simple cells**: V1-like orientation-selective receptive fields
- **Complex cells**: Phase-invariant feature integration
- **Inhibition patterns**: Local competition and predictive suppression
- **Learning**: Unsupervised STDP mimicking biological plasticity

This architecture reproduces cortical phenomena such as orientation tuning, surround suppression, and cross-orientation inhibition.

## References

For detailed architecture documentation, see [`PCL_Python_Analysis/docs/PCL_Network_Guide.md`](PCL_Python_Analysis/docs/PCL_Network_Guide.md).
