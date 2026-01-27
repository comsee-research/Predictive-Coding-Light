# PCL Parameters Reference Guide

This document provides a comprehensive reference for all parameters used in the Predictive Coding Light (PCL) C++ core implementation.

## Table of Contents

1. [Network Configuration Parameters](#network-configuration-parameters)
2. [Simple Neuron (SimpleCell) Parameters](#simple-neuron-simplecell-parameters)
3. [Complex Neuron (ComplexCell) Parameters](#complex-neuron-complexcell-parameters)
4. [Parameter Units](#parameter-units)
5. [Configuration File Structure](#configuration-file-structure)

---

## Network Configuration Parameters

These parameters are defined in `network_config.json` and control the overall network architecture.

### Network Structure Parameters

#### `nbCameras`
- **Type**: Integer
- **Description**: Number of camera inputs to the network. For most datasets (DVS-Gesture, Natural Images), this is typically 1. Multi-camera setups require appropriate event file formatting with camera IDs.
- **Example**: `1`
- **Location**: Network config

#### `neuron1Synapses`
- **Type**: Integer
- **Description**: Number of synaptic connections for first-layer neurons. Used to configure synaptic delays if `SYNAPSE_DELAY > 0`.
- **Example**: `1`
- **Location**: Network config

#### `sharingType`
- **Type**: String
- **Valid Values**: `"patch"`, `"none"`
- **Description**: Weight sharing strategy across neurons
  - `"patch"`: Neurons in the same layer share weight matrices (convolutional-like behavior for translation invariance)
  - `"none"`: Each neuron has independent weights
- **Example**: `"patch"`
- **Location**: Network config

#### `vfWidth`
- **Type**: Integer
- **Description**: Width of the visual field in pixels. Defines the horizontal extent of the input space that the network processes.
- **Example**: `346`, `66`
- **Location**: Network config

#### `vfHeight`
- **Type**: Integer
- **Description**: Height of the visual field in pixels. Defines the vertical extent of the input space that the network processes.
- **Example**: `260`, `66`
- **Location**: Network config

#### `measurementInterval`
- **Type**: Double (milliseconds)
- **Description**: Time interval (in ms) for network state measurements and statistics collection. Internal conversion: multiplied by 1000 to convert to microseconds (µs).
- **Example**: `100` (100 ms = 100,000 µs)
- **Location**: Network config

#### `neuronInhibitionRange`
- **Type**: Array of integers `[range_x, range_y]`
- **Description**: Spatial range for lateral dynamic inhibition connections. Defines the neighborhood size (in neuron grid coordinates) within which lateral inhibition is applied. A range of `[4, 4]` means inhibition extends ±4 neurons in x and y directions.
- **Example**: `[4, 4]`
- **Location**: Network config

### Layer-Specific Parameters

These parameters are arrays where each element corresponds to a layer in the network.

#### `neuronType`
- **Type**: Array of strings
- **Valid Values**: `"SimpleCell"`, `"ComplexCell"`
- **Description**: Type of neurons in each layer
  - `"SimpleCell"`: First-layer neurons that process raw events and learn input features via STDP
  - `"ComplexCell"`: Higher-layer neurons that integrate simple cell responses
- **Example**: `["SimpleCell", "ComplexCell"]`
- **Location**: Network config

#### `layerInhibitions`
- **Type**: Array of arrays of strings
- **Valid Values**: `"local"`, `"lateral"`, `"topdown"`
- **Description**: Types of inhibition active in each layer
  - `"local"`: Winner-take-all inhibition between neurons at the same spatial location (shared position, different feature channels)
  - `"lateral"`: Spatial competition within a neighborhood defined by `neuronInhibitionRange`
  - `"topdown"`: Predictive inhibition from higher layers to lower layers
- **Example**: `[["local", "lateral"], ["local", "lateral", "topdown"]]`
- **Location**: Network config
- **Notes**: Layer 0 typically cannot have `"topdown"` inhibition as there is no lower layer

#### `interLayerConnections`
- **Type**: Array of arrays of integers
- **Description**: Defines which lower layers each layer connects to. `-1` indicates no inter-layer connection (input layer). Layer n with value `[i, j]` receives input from layers i and j.
- **Example**: `[[-1], [0]]` - Layer 0 receives raw input, Layer 1 receives from Layer 0
- **Location**: Network config

#### `layerPatches`
- **Type**: Array of arrays of arrays `[[[x_offset], [y_offset], [z_offset]]]`
- **Description**: Offset for patch-based processing. Typically `[[[0], [0], [0]]]` for standard configurations. Used to define spatial shifts when implementing patching strategies.
- **Example**: `[[[0], [0], [0]], [[0], [0], [0]]]`
- **Location**: Network config

#### `layerSizes`
- **Type**: Array of arrays `[[width, height, features]]`
- **Description**: Dimensions of each layer in the network
  - `width`: Number of neurons along x-axis
  - `height`: Number of neurons along y-axis
  - `features`: Number of feature channels (depth)
- **Example**: `[[9, 9, 64], [6, 6, 32]]` - Layer 0: 9×9 grid with 64 features per location; Layer 1: 6×6 grid with 32 features
- **Location**: Network config

#### `neuronSizes`
- **Type**: Array of arrays of arrays `[[[rf_width, rf_height, rf_depth]]]`
- **Description**: Receptive field (RF) dimensions for neurons in each layer
  - For layer 0: RF size in input space (e.g., `[[10, 10, 1]]` means 10×10 pixel RF)
  - For higher layers: RF dimensions in terms of neurons from previous layer (e.g., `[[4, 4, 64]]` means pooling over 4×4 neurons with 64 features each)
- **Example**: `[[[10, 10, 1]], [[4, 4, 64]]]`
- **Location**: Network config

#### `neuronOverlap`
- **Type**: Array of arrays `[[overlap_x, overlap_y, overlap_z]]`
- **Description**: Overlap between adjacent neurons' receptive fields. Determines the stride of the receptive fields.
  - Higher overlap → more spatial continuity, more neurons covering similar regions
  - Overlap of `[3, 3, 0]` with RF size `[10, 10]` means stride of 7 pixels (10-3=7)
- **Example**: `[[3, 3, 0], [3, 3, 0]]`
- **Location**: Network config
- **Notes**: Z-overlap (depth) is typically 0 for standard configurations

---

## Simple Neuron (SimpleCell) Parameters

These parameters are defined in `simple_cell_config.json` and control the behavior of first-layer neurons.

### Membrane Dynamics Parameters

#### `TAU_M`
- **Type**: Double (milliseconds)
- **Description**: Membrane time constant. Controls the exponential decay rate of the neuron's membrane potential. Smaller values → faster decay, less temporal integration. Larger values → slower decay, more temporal integration.
- **Formula**: `V(t) = V(t₀) × exp(-(t - t₀) / TAU_M)`
- **Example**: `18` ms
- **Typical Range**: 10-50 ms
- **Unit Conversion**: Multiplied by 1000 internally to convert to µs

#### `VTHRESH`
- **Type**: Double (millivolts)
- **Description**: Spike threshold. When membrane potential exceeds this value, the neuron fires a spike. Can be adapted over time based on `TARGET_SPIKE_RATE`.
- **Example**: `10`, `30`
- **Typical Range**: 3-30 mV
- **Location**: Simple cell config

#### `VRESET`
- **Type**: Double (millivolts)
- **Description**: Reset potential. The membrane potential is set to this value immediately after a spike. Typically negative to create a hyperpolarized state.
- **Example**: `-10`
- **Typical Range**: -20 to 0 mV
- **Location**: Simple cell config

### STDP Learning Parameters

#### `TAU_LTP`
- **Type**: Double (milliseconds)
- **Description**: Time constant for Long-Term Potentiation (LTP). Controls the temporal window for strengthening synapses. Pre-synaptic events that occur before a post-synaptic spike within this window cause weight increases.
- **Formula**: `ΔLTP ∝ exp(-(t_spike - t_event) / TAU_LTP)`
- **Example**: `7` ms
- **Typical Range**: 5-40 ms
- **Unit Conversion**: Multiplied by 1000 internally to convert to µs

#### `TAU_LTD`
- **Type**: Double (milliseconds)
- **Description**: Time constant for Long-Term Depression (LTD). Controls the temporal window for weakening synapses. Pre-synaptic events that occur after a post-synaptic spike within this window cause weight decreases.
- **Formula**: `ΔLTD ∝ exp(-(t_event - t_last_spike) / TAU_LTD)`
- **Example**: `7` ms
- **Typical Range**: 5-40 ms
- **Unit Conversion**: Multiplied by 1000 internally to convert to µs

#### `ETA_LTP`
- **Type**: Double
- **Description**: Learning rate for excitatory synaptic potentiation. Controls the magnitude of weight increases during LTP. Combined with `NORM_FACTOR` to get the actual update magnitude.
- **Formula**: `Δw_LTP = softbound × decay × ETA_LTP × NORM_FACTOR × exp(-Δt / TAU_LTP)`
- **Example**: `0.000204`, `0.00077`
- **Typical Range**: 0.0001 - 0.001
- **Location**: Simple cell config

#### `ETA_LTD`
- **Type**: Double (typically negative)
- **Description**: Learning rate for excitatory synaptic depression. Controls the magnitude of weight decreases during LTD. Should be negative to reduce weights.
- **Formula**: `Δw_LTD = softbound × ETA_LTD × NORM_FACTOR × exp(-Δt / TAU_LTD)`
- **Example**: `-0.000204`, `-0.00021`
- **Typical Range**: -0.001 to -0.0001
- **Location**: Simple cell config

#### `ETA_ILTP`
- **Type**: Double
- **Description**: Learning rate for inhibitory synaptic potentiation. Controls strengthening of inhibitory connections. Used for plastic lateral and top-down inhibitory weights.
- **Formula**: `Δw_inhib_LTP = softbound × ETA_ILTP × INH_FACTOR × exp(-Δt / TAU_LTP)`
- **Example**: `0.33`, `0.0004600`
- **Typical Range**: 0.0001 - 1.0
- **Location**: Simple cell config

#### `ETA_ILTD`
- **Type**: Double (typically negative)
- **Description**: Learning rate for inhibitory synaptic depression. Controls weakening of inhibitory connections.
- **Formula**: `Δw_inhib_LTD = softbound × ETA_ILTD × INH_FACTOR × exp(-Δt / TAU_LTD)`
- **Example**: `-0.33`, `-0.0004600`
- **Typical Range**: -1.0 to -0.0001
- **Location**: Simple cell config

#### `NORM_FACTOR`
- **Type**: Double
- **Description**: Normalization factor for excitatory weights. After STDP updates, all excitatory weights are normalized to have an L1 norm equal to this value. Maintains stable weight magnitudes and prevents runaway potentiation.
- **Example**: `50`, `1000`
- **Typical Range**: 10-1000
- **Location**: Simple cell config

#### `LATERAL_NORM_FACTOR`
- **Type**: Double
- **Description**: Normalization factor for lateral inhibitory weights. Controls the strength of lateral dynamic inhibition between neurons in the same layer.
- **Example**: `6500`
- **Typical Range**: 1000-10000
- **Location**: Simple cell config

#### `TOPDOWN_NORM_FACTOR`
- **Type**: Double
- **Description**: Normalization factor for top-down inhibitory weights. Controls the strength of predictive feedback inhibition from higher layers.
- **Example**: `3500`, `2000`
- **Typical Range**: 1000-5000
- **Location**: Simple cell config

#### `ETA_INH`
- **Type**: Double
- **Description**: Normalization factor for plastic local inhibitory weights (winner-take-all inhibition). Controls the strength of local competition between neurons at the same spatial location.
- **Example**: `1500`
- **Typical Range**: 500-2000
- **Location**: Simple cell config

#### `STDP_LEARNING`
- **Type**: String
- **Valid Values**: `"all"`, `"excitatory"`, `"inhibitory"`, `"none"`
- **Description**: Controls which types of synaptic plasticity are active
  - `"all"`: Both excitatory and inhibitory weights are plastic
  - `"excitatory"`: Only excitatory weights adapt
  - `"inhibitory"`: Only inhibitory weights adapt
  - `"none"`: All weights are frozen (testing mode)
- **Example**: `"all"`
- **Location**: Simple cell config

#### `DECAY_RATE`
- **Type**: Double
- **Description**: Learning rate decay factor. Reduces learning rates over time to implement a learning schedule.
- **Formula**: `decay_multiplier = 1 / (1 + DECAY_RATE × count)`
- **Example**: `0` (no decay)
- **Typical Range**: 0 (no decay) to 0.01 (gradual decay)
- **Location**: Simple cell config

### Homeostatic and Adaptation Parameters

#### `TAU_RP`
- **Type**: Double (milliseconds)
- **Description**: Refractory period time constant. Controls the exponential decay of the hyperpolarizing refractory potential that follows each spike.
- **Formula**: `V_refrac(t) = DELTA_RP × exp(-(t - t_spike) / TAU_RP)`
- **Example**: `5` ms
- **Typical Range**: 2-10 ms
- **Unit Conversion**: Multiplied by 1000 internally to convert to µs

#### `DELTA_RP` (also called `ETA_RP` in config)
- **Type**: Double (millivolts)
- **Description**: Magnitude of refractory hyperpolarization. Immediately after spiking, this negative potential is added to prevent immediate re-spiking.
- **Example**: `10`
- **Typical Range**: 5-20 mV
- **Location**: Simple cell config

#### `TAU_SRA`
- **Type**: Double (milliseconds)
- **Description**: Spike Rate Adaptation time constant. Controls the decay of the adaptation potential that builds up with repeated spiking.
- **Formula**: `V_adapt(t) = V_adapt(t₀) × exp(-(t - t₀) / TAU_SRA)`
- **Example**: `100` ms
- **Typical Range**: 50-200 ms
- **Unit Conversion**: Multiplied by 1000 internally to convert to µs

#### `DELTA_SRA` (also called `ETA_SRA` in config)
- **Type**: Double (millivolts)
- **Description**: Magnitude of spike rate adaptation. This amount is added to the adaptation potential each time the neuron spikes, creating a progressively stronger hyperpolarizing force.
- **Example**: `0` (disabled in many configs)
- **Typical Range**: 0 (disabled) to 5 mV
- **Location**: Simple cell config

#### `TARGET_SPIKE_RATE`
- **Type**: Double (spikes/second)
- **Description**: Target firing rate for homeostatic threshold adaptation. The neuron's threshold adapts to maintain this average firing rate.
- **Example**: `0.75` (0.75 spikes/second)
- **Typical Range**: 0.1-2.0 Hz
- **Location**: Simple cell config

#### `MIN_THRESH`
- **Type**: Double (millivolts)
- **Description**: Minimum threshold value. Prevents adaptive threshold from dropping too low, which could cause excessive spiking.
- **Example**: `4`
- **Typical Range**: 1-10 mV
- **Location**: Simple cell config

#### `ETA_SR` (also called `ETA_TA` in config)
- **Type**: Double
- **Description**: Learning rate for spike rate-based threshold adaptation. Controls how quickly the threshold adapts to match `TARGET_SPIKE_RATE`.
- **Formula**: 
  - If rate > target: `threshold += ETA_SR × (1 - exp(target - rate))`
  - If rate < target: `threshold -= ETA_SR × (1 - exp(rate - target))`
- **Example**: `0` (disabled in many configs)
- **Typical Range**: 0 (disabled) to 0.1
- **Location**: Simple cell config

### Synaptic Delay and Tracking Parameters

#### `SYNAPSE_DELAY`
- **Type**: Integer (microseconds)
- **Description**: Synaptic transmission delay. If non-zero, creates multiple delayed versions of input events. The number of delayed versions equals `neuron1Synapses`.
- **Example**: `0` (no delay)
- **Typical Range**: 0 (no delay) to 5000 µs
- **Location**: Simple cell config

#### `TRACKING`
- **Type**: String
- **Valid Values**: `"none"`, `"partial"`, `"full"`
- **Description**: Controls whether and how neuron activity is tracked
  - `"none"`: No tracking (fastest)
  - `"partial"`: Track spike times for specified neurons (via `POTENTIAL_TRACK`)
  - `"full"`: Track all neuron activities (slowest, for debugging)
- **Example**: `"none"`
- **Location**: Simple cell config

#### `POTENTIAL_TRACK`
- **Type**: Array of integers `[x, y]`
- **Description**: Grid coordinates of the neuron(s) to track when `TRACKING = "partial"`. Allows monitoring specific neurons' membrane potentials and spike trains.
- **Example**: `[4, 4]`
- **Location**: Simple cell config

---

## Complex Neuron (ComplexCell) Parameters

These parameters are defined in `complex_cell_config.json` and control the behavior of higher-layer neurons. Complex neurons have fewer parameters than simple neurons since they don't process raw events.

### Membrane Dynamics Parameters

#### `TAU_M`
- **Type**: Double (milliseconds)
- **Description**: Membrane time constant. Same as for simple neurons but typically longer for complex cells to integrate over more inputs.
- **Example**: `50` ms
- **Typical Range**: 30-100 ms
- **Unit Conversion**: Multiplied by 1000 internally to convert to µs

#### `VTHRESH`
- **Type**: Double (millivolts)
- **Description**: Spike threshold for complex neurons. Often lower than simple cell threshold since they integrate pre-processed spike inputs.
- **Example**: `3`
- **Typical Range**: 1-10 mV
- **Location**: Complex cell config

#### `VRESET`
- **Type**: Double (millivolts)
- **Description**: Reset potential after spiking.
- **Example**: `-10`
- **Typical Range**: -20 to 0 mV
- **Location**: Complex cell config

### STDP Learning Parameters

#### `TAU_LTP`
- **Type**: Double (milliseconds)
- **Description**: Time constant for LTP in complex neurons.
- **Example**: `40` ms
- **Typical Range**: 20-100 ms
- **Unit Conversion**: Multiplied by 1000 internally to convert to µs

#### `TAU_LTD`
- **Type**: Double (milliseconds)
- **Description**: Time constant for LTD in complex neurons.
- **Example**: `40` ms
- **Typical Range**: 20-100 ms
- **Unit Conversion**: Multiplied by 1000 internally to convert to µs

#### `ETA_LTP`
- **Type**: Double
- **Description**: Learning rate for excitatory synaptic potentiation. Typically larger than for simple cells.
- **Example**: `0.08`, `0.002`
- **Typical Range**: 0.001 - 0.1
- **Location**: Complex cell config

#### `ETA_LTD`
- **Type**: Double
- **Description**: Learning rate for excitatory synaptic depression. Can be positive or negative depending on implementation.
- **Example**: `-0.08`, `0.002`
- **Typical Range**: -0.1 to 0.1
- **Location**: Complex cell config

#### `ETA_ILTP`
- **Type**: Double
- **Description**: Learning rate for inhibitory synaptic potentiation in complex neurons.
- **Example**: `0.25`, `0.008`
- **Typical Range**: 0.001 - 1.0
- **Location**: Complex cell config

#### `ETA_ILTD`
- **Type**: Double (typically negative)
- **Description**: Learning rate for inhibitory synaptic depression in complex neurons.
- **Example**: `-0.25`, `-0.008`
- **Typical Range**: -1.0 to -0.001
- **Location**: Complex cell config

#### `NORM_FACTOR`
- **Type**: Double
- **Description**: Normalization factor for excitatory weights. Typically larger than simple cell norm factor.
- **Example**: `1000`
- **Typical Range**: 100-2000
- **Location**: Complex cell config

#### `ETA_INH`
- **Type**: Double
- **Description**: Normalization factor for inhibitory weights in complex neurons.
- **Example**: `600`
- **Typical Range**: 200-1000
- **Location**: Complex cell config

#### `STDP_LEARNING`
- **Type**: String
- **Valid Values**: `"all"`, `"excitatory"`, `"inhibitory"`, `"none"`
- **Description**: Controls which types of synaptic plasticity are active.
- **Example**: `"all"`
- **Location**: Complex cell config

#### `DECAY_RATE`
- **Type**: Double
- **Description**: Learning rate decay factor for complex neurons.
- **Example**: `0`
- **Typical Range**: 0 to 0.01
- **Location**: Complex cell config

### Homeostatic Parameters

#### `TAU_RP`
- **Type**: Double (milliseconds)
- **Description**: Refractory period time constant.
- **Example**: `5` ms
- **Typical Range**: 2-10 ms
- **Unit Conversion**: Multiplied by 1000 internally to convert to µs

#### `DELTA_RP` (also called `ETA_RP` in config)
- **Type**: Double (millivolts)
- **Description**: Magnitude of refractory hyperpolarization.
- **Example**: `5`
- **Typical Range**: 2-15 mV
- **Location**: Complex cell config

### Tracking Parameters

#### `TRACKING`
- **Type**: String
- **Valid Values**: `"none"`, `"partial"`, `"full"`
- **Description**: Controls activity tracking.
- **Example**: `"none"`
- **Location**: Complex cell config

#### `POTENTIAL_TRACK`
- **Type**: Array of integers `[x, y]`
- **Description**: Coordinates of neurons to track.
- **Example**: `[2, 3]`
- **Location**: Complex cell config

---

## Parameter Units

### Time Units
All time-related parameters use **milliseconds** in the JSON configuration files but are converted internally:
- Network measurement intervals: ms → µs (× 1000)
- Neuron time constants: ms → µs (× 1000)
- Synaptic delays: directly in µs

### Voltage Units
All voltage-related parameters use **millivolts (mV)**:
- Thresholds (VTHRESH)
- Reset potentials (VRESET)
- Refractory potentials (DELTA_RP)
- Adaptation potentials (DELTA_SRA)

### Learning Rates
- Dimensionless (scaled by normalization factors)
- ETA parameters are typically in range [0.0001, 1.0]
- Can be negative for depression (LTD)

### Spatial Units
- Coordinates and sizes: integer pixel/neuron units
- Visual field dimensions: pixels
- Layer dimensions: neuron grid coordinates

---

## Configuration File Structure

A complete PCL network requires three JSON configuration files:

### 1. `network_config.json`
```json
{
    "nbCameras": 1,
    "neuron1Synapses": 1,
    "sharingType": "patch",
    "vfWidth": 66,
    "vfHeight": 66,
    "measurementInterval": 100,
    "neuronInhibitionRange": [4, 4],
    "neuronType": ["SimpleCell", "ComplexCell"],
    "layerInhibitions": [["local", "lateral"], ["local", "lateral", "topdown"]],
    "interLayerConnections": [[-1], [0]],
    "layerPatches": [[[0], [0], [0]], [[0], [0], [0]]],
    "layerSizes": [[9, 9, 64], [6, 6, 32]],
    "neuronSizes": [[[10, 10, 1]], [[4, 4, 64]]],
    "neuronOverlap": [[3, 3, 0], [3, 3, 0]]
}
```

### 2. `simple_cell_config.json`
```json
{
    "VTHRESH": 10,
    "VRESET": -10,
    "TAU_M": 18,
    "TAU_LTP": 7,
    "TAU_LTD": 7,
    "TAU_RP": 5,
    "TAU_SRA": 100,
    "ETA_LTP": 0.000204,
    "ETA_LTD": -0.000204,
    "ETA_ILTP": 0.33,
    "ETA_ILTD": -0.33,
    "ETA_INH": 1500,
    "ETA_RP": 10,
    "ETA_SRA": 0,
    "ETA_TA": 0,
    "NORM_FACTOR": 50,
    "LATERAL_NORM_FACTOR": 6500,
    "TOPDOWN_NORM_FACTOR": 3500,
    "DECAY_RATE": 0,
    "TARGET_SPIKE_RATE": 0.75,
    "MIN_THRESH": 4,
    "SYNAPSE_DELAY": 0,
    "STDP_LEARNING": "all",
    "TRACKING": "none",
    "POTENTIAL_TRACK": [4, 4]
}
```

### 3. `complex_cell_config.json`
```json
{
    "VTHRESH": 3,
    "VRESET": -10,
    "TAU_M": 50,
    "TAU_LTP": 40,
    "TAU_LTD": 40,
    "TAU_RP": 5,
    "ETA_LTP": 0.08,
    "ETA_LTD": -0.08,
    "ETA_ILTP": 0.25,
    "ETA_ILTD": -0.25,
    "ETA_INH": 600,
    "ETA_RP": 5,
    "NORM_FACTOR": 1000,
    "DECAY_RATE": 0,
    "STDP_LEARNING": "all",
    "TRACKING": "none",
    "POTENTIAL_TRACK": [2, 3]
}
```

---

## Parameter Tuning Guidelines

### For Excitatory Learning
1. **Increase selectivity**: Increase `NORM_FACTOR`, decrease `ETA_LTP/ETA_LTD`
2. **Faster learning**: Increase `ETA_LTP/ETA_LTD`, decrease `TAU_LTP/TAU_LTD`
3. **Broader temporal integration**: Increase `TAU_M`

### For Inhibition
1. **Stronger competition**: Increase `ETA_INH`, `LATERAL_NORM_FACTOR`, or `TOPDOWN_NORM_FACTOR`
2. **Adjust inhibition plasticity**: Modify `ETA_ILTP/ETA_ILTD`
3. **Change inhibition types**: Add/remove from `layerInhibitions`

### For Network Architecture
1. **Larger receptive fields**: Increase values in `neuronSizes`
2. **More features**: Increase third dimension in `layerSizes`
3. **Denser coverage**: Increase values in `neuronOverlap`
4. **Wider inhibition**: Increase `neuronInhibitionRange`

### For Homeostasis
1. **Target specific firing rate**: Set `TARGET_SPIKE_RATE` and `ETA_SR > 0`
2. **Reduce bursting**: Increase `DELTA_SRA` and `TAU_SRA`
3. **Stronger refractoriness**: Increase `DELTA_RP`

---

## Implementation Notes

### Internal Calculations

1. **Excitatory STDP Update**:
   ```cpp
   m_eLTP = ETA_LTP × NORM_FACTOR
   m_eLTD = ETA_LTD × NORM_FACTOR
   Δw_LTP = softbound × decay × m_eLTP × exp(-Δt / TAU_LTP)
   Δw_LTD = softbound × m_eLTD × exp(-Δt / TAU_LTD)
   ```

2. **Inhibitory STDP Update**:
   ```cpp
   // For local inhibition:
   m_iLTP = ETA_ILTP × ETA_INH
   m_iLTD = ETA_ILTD × ETA_INH
   
   // For lateral inhibition:
   m_pliLTP = ETA_ILTP × LATERAL_NORM_FACTOR
   m_pliLTD = ETA_ILTD × LATERAL_NORM_FACTOR
   
   // For top-down inhibition:
   m_ptiLTP = ETA_ILTP × TOPDOWN_NORM_FACTOR
   m_ptiLTD = ETA_ILTD × TOPDOWN_NORM_FACTOR
   ```

3. **Membrane Potential Decay**:
   ```cpp
   V(t) = V(t₀) × exp(-(t - t₀) / TAU_M)
   ```

4. **Refractory Potential**:
   ```cpp
   V_refrac(t) = DELTA_RP × exp(-(t - t_spike) / TAU_RP)
   ```

5. **Threshold Adaptation**:
   ```cpp
   if (spike_rate > TARGET_SPIKE_RATE):
       threshold += ETA_SR × (1 - exp(TARGET_SPIKE_RATE - spike_rate))
   else:
       threshold -= ETA_SR × (1 - exp(spike_rate - TARGET_SPIKE_RATE))
   
   threshold = max(threshold, MIN_THRESH)
   ```

### Soft Bounds in Weight Updates
The implementation uses soft-bound constraints on weight updates to prevent weights from exceeding maximum values or going negative:
- Maximum excitatory weight: `max_eWeight = 3`
- Maximum inhibitory weight: `max_iWeight = 50`
- Soft bound factor for excitation: `factor_eBound = 0.33`
- Soft bound factor for inhibition: `factor_iBound = 0.02`

These are hard-coded in the implementation and modify the STDP updates multiplicatively.

---

## References

For more information on the biological inspiration and architecture of the PCL network, see:
- [PROJECT.md](PROJECT.md) - Project overview
- [PCL_Python_Analysis/docs/PCL_Network_Guide.md](PCL_Python_Analysis/docs/PCL_Network_Guide.md) - Detailed architecture documentation
- Original Predictive Coding Light paper

---

**Document Version**: 1.0  
**Last Updated**: January 2026  
**Author**: Generated from PCL C++ codebase analysis
