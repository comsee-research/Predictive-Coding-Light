# Spike Recording in PCL

## Overview

**Yes, the PCL codebase fully supports recording output spikes from both Simple and Complex cells.** The spike recording system is implemented at the C++ core level and can be accessed through the Python interface for analysis and visualization.

## How Spike Recording Works

### 1. **Enabling Spike Recording**

Spike recording is controlled via the `TRACKING` parameter in the neuron configuration files:

**Configuration Files:**
- `configs/simple_cell_config.json` - for SimpleCell (first layer)
- `configs/complex_cell_config.json` - for ComplexCell (higher layers)

**TRACKING Parameter Options:**
- `"none"` - No spike recording (default, minimal memory usage)
- `"partial"` - Records spike timestamps for orientation tuning analysis
- `"all"` - Records all spike information (not fully implemented)

**Example configuration:**
```json
{
    "TRACKING": "partial",
    ...
}
```

### 2. **C++ Implementation**

#### Spike Storage Mechanism

**In SimpleNeuron** ([libs/network/src/neurons/SimpleNeuron.cpp](libs/network/src/neurons/SimpleNeuron.cpp)):
```cpp
inline void SimpleNeuron::spike(const size_t time) {
    m_vectorSpikingTime.push_back(time);
    m_lastSpikingTime = m_spikingTime;
    m_spikingTime = time;
    m_spike = true;
    ++m_spikeRateCounter;
    ++m_totalSpike;
    m_spikingPotential = m_potential;
    m_potential = m_conf.VRESET;
    
    if (m_conf.TRACKING == "partial") {
        m_trackingSpikeTrain.push_back(time - m_firstInput);
    }
}
```

**In ComplexNeuron** ([libs/network/src/neurons/ComplexNeuron.cpp](libs/network/src/neurons/ComplexNeuron.cpp)):
```cpp
inline void ComplexNeuron::spike(const size_t time) {
    m_vectorSpikingTime.push_back(time);
    m_lastSpikingTime = m_spikingTime;
    m_spikingTime = time;
    m_spike = true;
    ++m_spikeRateCounter;
    ++m_totalSpike;
    m_spikingPotential = m_potential;
    m_potential = m_conf.VRESET;
    
    if (m_conf.TRACKING == "partial") {
        m_trackingSpikeTrain.push_back(time - m_firstInput);
    }
}
```

#### Data Structure

- **Storage:** `std::vector<size_t> m_trackingSpikeTrain` - stores spike timestamps
- **Timestamps:** Recorded relative to the first input (`time - m_firstInput`)
- **Units:** Microseconds (µs)
- **Access:** Via `getTrackingSpikeTrain()` method

### 3. **Saving Spikes to Disk**

#### JSON Format

Spikes are automatically saved when calling `saveStatistics()` on the network:

**C++ Side** ([libs/network/src/neurons/Neuron.cpp](libs/network/src/neurons/Neuron.cpp), line 488):
```cpp
void Neuron::writeJson(nlohmann::json &state) {
    // ... other fields ...
    state["spike_train"] = m_trackingSpikeTrain;
}
```

**File Structure:**
```
network_path/
└── statistics/
    └── {layer_id}/
        └── {folder_name}{simulation}/
            └── {sequence}/
                ├── 0.json      # Neuron 0 statistics
                ├── 1.json      # Neuron 1 statistics
                └── ...
```

**JSON File Content Example:**
```json
{
    "spike_train": [125000, 250300, 375150, ...],
    "count_spike": 127,
    "potential_train": [[2.5, 125000], [3.1, 125500], ...],
    "threshold": 10.0,
    ...
}
```

### 4. **Recording Spikes During Execution**

To record spikes, use the C++ NetworkHandle or SurroundSuppression classes:

**Method 1: Via NetworkHandle** ([libs/network/include/network/NetworkHandle.hpp](libs/network/include/network/NetworkHandle.hpp)):
```cpp
NetworkHandle network("/path/to/network/");
network.setEventPath("/path/to/events.h5");

// Process events
Events events;
while (network.loadEvents(events, num_passes)) {
    network.feedEvents(events);
}

// Save statistics with spikes
network.saveStatistics(simulation_id, sequence_id, "folder_name/");
```

**Method 2: Via SurroundSuppression** ([libs/network/src/SurroundSuppression.cpp](libs/network/src/SurroundSuppression.cpp), line 1221):
```cpp
void SurroundSuppression::recordSpikes(const std::string &vectorOfPath) {
    for (const auto & frame : std::filesystem::directory_iterator{vectorOfPath}) {
        m_network.setEventPath(frame.path().string());
        Events events;
        while (m_network.loadEvents(events, 1)) {
            m_network.feedEvents(events);
        }
        m_network.saveStatistics(sim_id, 1, "folder_name/");
    }
}
```

### 5. **Python Interface - Loading and Analyzing Spikes**

#### Loading Spike Trains

**SpikingNetwork Class** ([PCL_Python_Analysis/src/spiking_network/network/neuvisys.py](PCL_Python_Analysis/src/spiking_network/network/neuvisys.py)):

```python
from src.spiking_network.network.neuvisys import SpikingNetwork

# Load network with spike trains
network = SpikingNetwork("/path/to/network/", loading=True)

# Access spikes per layer
spikes_layer_0 = network.spikes[0]  # Simple cells
spikes_layer_1 = network.spikes[1]  # Complex cells

# Spike format: 2D numpy array
# Shape: (num_neurons, max_spike_times)
# Values: spike timestamps in microseconds (0 = no spike)
```

#### Spike Data Format in Python

After loading, spikes are stored as:
```python
# Each layer: numpy array of shape (N_neurons, max_spikes)
network.spikes[layer_id][neuron_id]  # Array of spike times

# Example:
# neuron 5 in layer 0:
>>> network.spikes[0][5]
array([125000, 250300, 375150, 0, 0, ...])  # Padded with zeros
```

#### Loading Statistics with Spikes

For detailed analysis including spike trains from specific simulations:

```python
# Load statistics for a specific simulation
network.load_statistics_standard(
    folder_name="experiment_01",
    layer_id=0,
    simulation=0
)

# Access loaded statistics
for sequence in network.stats:
    for layer_data in sequence.values():
        for layer in layer_data:
            for neuron_stats in layer.values():
                # Each neuron's data is a list of dicts
                amount_events = neuron_stats[0]["amount_of_events"]
                potential_train = neuron_stats[1]["potential_train"]
                # Note: spike_train is loaded from neuron.params["spike_train"]
```

### 6. **Analysis and Visualization**

#### Spike Train Analysis

**Using the analysis module** ([PCL_Python_Analysis/src/spiking_network/analysis/spike_train.py](PCL_Python_Analysis/src/spiking_network/analysis/spike_train.py)):

```python
from src.spiking_network.analysis.spike_train import *
import quantities as pq

# Convert to spike train format
spike_trains = spike_trains(network.spikes[0])  # Convert to SpikeTrain objects

# Generate visualizations
raster_plot(spike_trains, layer=0, path="./results/")
event_plot(spike_trains, layer=0, path="./results/")
time_histogram(spike_trains, layer=0, path="./results/")
spike_rate_histogram(network.spikes[0], layer=0, path="./results/")
```

#### Spike Counting and Statistics

```python
# Count spikes per neuron
spike_counts = np.count_nonzero(network.spikes[0], axis=1)

# Calculate spike rates (spikes/second)
time_duration = np.max(network.spikes[0]) * 1e-6  # Convert µs to seconds
spike_rates = spike_counts / time_duration

# Get mean and std
mean_rate = np.mean(spike_rates)
std_rate = np.std(spike_rates)

# Using built-in method
mean_rate, std_rate = network.spike_rate()
```

#### Response Analysis

For orientation tuning and other response properties:

```python
from src.spiking_network.analysis.network_statistics import *

# Compute tuning curves from spike responses
tuningCurvesOutput(
    spinet=network,
    save_folder="./tuning/",
    angles=np.arange(0, 180, 22.5),
    surr_angles=np.arange(0, 180, 22.5),
    neuron_id=0,
    fqcy=0.05,
    layer_id=0,
    n_simulation=10
)
```

### 7. **Output File Formats**

#### JSON Files (Individual Neurons)
- **Location:** `{network_path}/statistics/{layer}/{folder}{sim}/{seq}/{neuron_id}.json`
- **Format:** JSON
- **Content:** Complete neuron state including spike train
- **Spike field:** `"spike_train": [timestamp1, timestamp2, ...]`

#### NPY Files (Analysis Results)
- **Location:** Various analysis-specific paths
- **Format:** NumPy binary (`.npy`)
- **Examples:**
  - Spike counts: `spikes_avg.npy`
  - Standard errors: `spikes_stderr.npy`
  - Rotation responses: `rotation_response.npy`

### 8. **Practical Usage Example**

#### Complete Workflow

```python
from src.spiking_network.planning.network_planner import launch_neuvisys_multi_pass
from src.spiking_network.network.neuvisys import SpikingNetwork
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Enable spike tracking in config
# Edit configs/simple_cell_config.json: "TRACKING": "partial"

# Step 2: Train/run network (automatically records spikes)
launch_neuvisys_multi_pass(
    exec_path="/path/to/build/bin/neuvisys-exe",
    network_path="./my_network/",
    event_file="data/events/shapes.h5",
    nb_pass=5
)

# Step 3: Load network with spikes
network = SpikingNetwork("./my_network/", loading=True)

# Step 4: Analyze spikes
layer_0_spikes = network.spikes[0]

# Get spike statistics per neuron
for neuron_id in range(layer_0_spikes.shape[0]):
    spike_times = layer_0_spikes[neuron_id]
    spike_times = spike_times[spike_times > 0]  # Remove padding zeros
    
    if len(spike_times) > 0:
        print(f"Neuron {neuron_id}:")
        print(f"  Total spikes: {len(spike_times)}")
        print(f"  First spike: {spike_times[0]} µs")
        print(f"  Last spike: {spike_times[-1]} µs")
        print(f"  Mean ISI: {np.mean(np.diff(spike_times)):.2f} µs")

# Step 5: Visualize spike raster plot
plt.figure(figsize=(12, 6))
for neuron_id in range(min(50, layer_0_spikes.shape[0])):
    spike_times = layer_0_spikes[neuron_id]
    spike_times = spike_times[spike_times > 0] / 1e6  # Convert to seconds
    plt.scatter(spike_times, [neuron_id] * len(spike_times), 
                s=1, c='black')
plt.xlabel('Time (s)')
plt.ylabel('Neuron ID')
plt.title('Spike Raster Plot - Layer 0')
plt.tight_layout()
plt.savefig('spike_raster.png', dpi=300)
```

## Summary

### ✅ Spike Recording Capabilities

1. **Both cell types supported:** Simple and Complex neurons
2. **Automatic recording:** Enabled via `TRACKING` config parameter
3. **Persistent storage:** Saved in JSON format per neuron
4. **Python access:** Full API for loading and analyzing spikes
5. **Temporal precision:** Microsecond-level timestamps
6. **Flexible output:** JSON (raw) and NPY (processed) formats

### 🔧 Key Parameters

- **Enable:** Set `"TRACKING": "partial"` in `simple_cell_config.json` and `complex_cell_config.json`
- **Access:** Via `network.spikes[layer_id][neuron_id]` in Python
- **Units:** Microseconds (µs) relative to first input
- **Storage:** `{network_path}/statistics/` directory structure

### 📊 Analysis Tools Available

- Raster plots and event plots
- Time histograms and spike rate distributions
- Orientation tuning curves
- Cross-orientation suppression analysis
- Temporal response patterns
- Spike-based feature classification (SVM)

### ⚠️ Important Notes

1. **Memory usage:** `"partial"` tracking is efficient; only spike times are stored
2. **Performance:** Minimal impact on simulation speed when using "partial"
3. **Default setting:** TRACKING is "none" by default (must be explicitly enabled)
4. **Persistence:** Spikes are saved only when `saveStatistics()` is called
5. **Reset:** Spike trains can be reset with `network.resetSTrain()` in C++
