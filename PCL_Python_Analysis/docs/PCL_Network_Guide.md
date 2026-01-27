# PCL Network Architecture Guide

This document provides detailed explanations of the Predictive Coding Light (PCL) network architecture, inhibition types, and configuration options.

---

## Paper Terminology vs Code Implementation

The paper describes **three types of inhibitory connections**. Here's how they map to the code:

---

### 📌 **1. Short-Ranging Lateral Inhibition** (Winner-Take-All)

**Paper Description:** "Units in a layer can inhibit other units that receive inputs from the same retinal location, in order to promote the learning of diverse features."

**Code Implementation:** `"local"` in `layerInhibitions`
- **C++ Function:** `lateralLocalInhibitionConnection()` in `SpikingNetwork.cpp` (line 450)
- **Connection Pattern:** Neurons at position `(x, y, z)` inhibit **all other neurons** at the **same spatial location** `(x, y)` but different depths `z`
- **Purpose:** Competition between different feature detectors viewing the same retinal/input location
- **Weight Files:** `Nlli.npy` (lateral local inhibition)
- **Code Logic:**
  ```cpp
  // Connect to all neurons at same (x,y) but different z
  for (size_t z = 0; z < layerSizes[2]; ++z) {
      if (z != neuron.getPos().z()) {
          neuron.addLateralLocalInhibitionConnections(...);
      }
  }
  ```

---

### 📌 **2. Long-Ranging Lateral Inhibition** (Spatial Competition)

**Paper Description:** "The long-ranging lateral inhibitory connections affect distant units that receive inputs from different retinal locations."

**Code Implementation:** `"lateral"` in `layerInhibitions`
- **C++ Function:** `lateralDynamicInhibitionConnection()` in `SpikingNetwork.cpp` (line 409)
- **Connection Pattern:** Neurons inhibit **spatially nearby neurons** within a defined range, across **all depths**
- **Range Configuration:** `neuronInhibitionRange: [x_range, y_range]` (e.g., `[4, 4]` = ±4 neurons in each direction)
- **Purpose:** Broader spatial competition to enforce sparse coding across the visual field
- **Weight Files:** `Nli.npy` (lateral inhibition), `Nle.npy` (lateral excitation - created automatically)
- **Code Logic:**
  ```cpp
  // Connect to neighbors within inhibition range
  for (x in range(neuron.x - range_x, neuron.x + range_x + 1))
      for (y in range(neuron.y - range_y, neuron.y + range_y + 1))
          for (z in all depths)
              if (x,y) != neuron position:
                  neuron.addLateralDynamicInhibitionConnections(...);
  ```

---

### 📌 **3. Top-Down Inhibition** (Predictive Feedback)

**Paper Description:** "The top-down inhibitory connections consist of feedback projections of units in a higher layer to the units of their input zone in the lower layer."

**Code Implementation:** `"topdown"` in `layerInhibitions`
- **C++ Function:** `topDownConnection()` in `SpikingNetwork.cpp` (line 368)
- **Connection Pattern:** Neurons in **layer N+1** send feedback to neurons in **layer N** within their receptive field
- **Purpose:** Predictive coding - higher layers suppress lower-layer activity that they can explain/predict
- **Weight Files:** `Ntdi.npy` (top-down inhibition), `Ntde.npy` (top-down excitation)
- **Code Logic:**
  ```cpp
  // When building layer N connections to layer N-1
  if ("topdown" in inhibitions) {
      neuron.addTopDownDynamicInhibitionConnection(prevLayerNeuron);
      prevLayerNeuron.initTopDownDynamicInhibitionWeights(neuron.getIndex());
  }
  ```

---

## 🔗 Summary Mapping Table

| Paper Term | Config Value | Code Function | Connection Pattern | Weight Files |
|------------|--------------|---------------|-------------------|--------------|
| **Short-Ranging Lateral** | `"local"` | `lateralLocalInhibitionConnection()` | Same (x,y), different z | `Nlli.npy` |
| **Long-Ranging Lateral** | `"lateral"` | `lateralDynamicInhibitionConnection()` | Within spatial range, all z | `Nli.npy`, `Nle.npy` |
| **Top-Down** | `"topdown"` | `topDownConnection()` | Layer N+1 → Layer N (receptive field) | `Ntdi.npy`, `Ntde.npy` |

---

## 💡 Key Insights

1. **"local" is always enabled** - it's the fundamental Winner-Take-All mechanism at each location
2. **"lateral" is optional** - adds broader spatial competition (increases computational cost)
3. **"topdown" requires multiple layers** - implements predictive coding feedback
4. **All three can be combined** - the full PCL model uses all three types simultaneously
5. **Excitation weights created automatically** - both lateral and top-down create excitation weights alongside inhibition

---

## 📖 Default Configuration

The default network uses **only "local" inhibition** (short-ranging):
```json
"layerInhibitions": [["local"], ["local"]]
```

To match the full paper description, enable all three:
```json
"layerInhibitions": [["local", "lateral"], ["local", "lateral", "topdown"]]
```

---

## Configuration Parameters Reference

Below is a comprehensive explanation of all parameters in the Simple Cell and Complex Cell configuration files, based on analysis of the C++ codebase.

---

### 🧠 **Membrane Dynamics Parameters**

#### **VTHRESH** (Voltage Threshold)
- **Role**: Spike threshold - neuron fires when membrane potential exceeds this value
- **Usage**: Compared against membrane potential in `membraneUpdate()`
- **Simple cells**: 30 mV (higher, more selective)
- **Complex cells**: 3 mV (lower, more integrative)
- **Effect**: Higher values = fewer spikes, more selective responses

#### **VRESET** (Reset Voltage)
- **Role**: Membrane potential value after a spike
- **Usage**: `m_potential = m_conf.VRESET` after spike in `spike()`
- **Both cells**: -10 mV
- **Effect**: Determines how quickly neuron can fire again

#### **TAU_M** (Membrane Time Constant)
- **Role**: Controls membrane potential decay rate
- **Usage**: `exp(-Δt / TAU_M)` in `potentialDecay()`
- **Simple cells**: 18 ms
- **Complex cells**: 50 ms
- **Effect**: Larger = slower decay, longer temporal integration window

---

### 🔄 **STDP Learning Parameters**

#### **ETA_LTP** (LTP Learning Rate)
- **Role**: Controls weight increase when pre-synaptic event precedes post-synaptic spike
- **Usage**: `w += softbound * ETA_LTP * exp(-Δt / TAU_LTP)`
- **Simple cells**: 0.00077 (feedforward excitation)
- **Complex cells**: 0.002 (inter-layer connections)
- **Effect**: Higher = faster weight growth during LTP

#### **ETA_LTD** (LTD Learning Rate)
- **Role**: Controls weight decrease when post-synaptic spike precedes pre-synaptic event
- **Usage**: `w += softbound * ETA_LTD * exp(-Δt / TAU_LTD)`
- **Simple cells**: -0.00021 (negative = depression)
- **Complex cells**: -0.002 (negative = depression)
- **Effect**: More negative = stronger weight depression

#### **TAU_LTP** / **TAU_LTD** (STDP Time Constants)
- **Role**: Control temporal window for spike-timing dependent plasticity
- **Usage**: Exponential decay in STDP rule
- **Simple cells**: 7 ms
- **Complex cells**: 40 ms
- **Effect**: Larger = wider temporal window for learning

#### **ETA_ILTP** / **ETA_ILTD** (Inhibitory STDP Rates)
- **Role**: Learning rates for lateral and topdown inhibitory connections
- **Usage**: 
  - Lateral: `m_pliLTP = ETA_ILTP * LATERAL_NORM_FACTOR`
  - Topdown: `m_ptiLTP = ETA_ILTP * TOPDOWN_NORM_FACTOR`
- **Simple cells**: ±0.00046
- **Complex cells**: ±0.008
- **Effect**: Controls how inhibitory weights adapt

---

### ⚖️ **Weight Normalization Parameters** ⭐

These parameters directly control weight bounds and normalization:

#### **NORM_FACTOR** 
- **Role**: **Target L1 norm for excitatory (feedforward) weights**
- **Usage**: `m_sharedWeights.normalizeL1Norm(NORM_FACTOR)`
- **Implementation**: `element *= NORM_FACTOR / L1_norm`
- **Simple cells**: 50
- **Complex cells**: 1000
- **Effect**: All excitatory weights are rescaled so their sum equals this value
- **💡 Key**: This is the **primary control for excitatory weight strength**

#### **LATERAL_NORM_FACTOR** (Simple cells only)
- **Role**: **Target L1 norm for lateral inhibition/excitation weights**
- **Usage**: Combined with ETA_ILTP/ILTD to set inhibitory learning rates
- **Value**: 6500
- **Effect**: Much larger than NORM_FACTOR → lateral inhibition can be very strong
- **💡 Key**: Controls the **strength scale of lateral interactions**

#### **TOPDOWN_NORM_FACTOR** (Simple cells only)
- **Role**: **Target L1 norm for topdown inhibition/excitation weights**
- **Usage**: `m_topDownInhibitionWeights.normalizeL1Norm(TOPDOWN_NORM_FACTOR)`
- **Value**: 2000
- **Effect**: Intermediate strength between feedforward and lateral
- **💡 Key**: Controls the **strength scale of feedback from higher layers**

#### **ETA_INH**
- **Role**: **Target L1 norm for local inhibitory weights**
- **Usage**: `m_sharedInhibWeights.normalizeL1Norm(ETA_INH)`
- **Simple cells**: 1500
- **Complex cells**: 600
- **Effect**: Local inhibition strength constraint
- **💡 Key**: Controls **local competitive inhibition strength**

---

### 🔒 **Soft Weight Bounds (Hardcoded in C++)**

The C++ code uses soft bounds to prevent weights from growing unbounded:

```cpp
// Excitatory weights (feedforward)
double max_eWeight = 3;      // Simple: max individual weight
double factor_eBound = 0.33; // Soft bound factor
double softboundLTP = (max_eWeight - wj) * factor_eBound;
double softboundLTD = (wj) * factor_eBound;
```

**Effect**: 
- LTP slows down as weight approaches `max_eWeight` (3)
- LTD slows down as weight approaches 0
- Prevents runaway growth while maintaining plasticity
- **After each update, weights are normalized using NORM_FACTOR**

---

### 🚫 **Refractory Period Parameters**

#### **TAU_RP** (Refractory Time Constant)
- **Role**: Duration of refractory period after spike
- **Usage**: `exp(-Δt / TAU_RP)` in `refractoryPotential()`
- **Both cells**: 5 ms
- **Effect**: Prevents rapid successive spikes

#### **ETA_RP** (Refractory Increment)
- **Role**: Increase in refractory potential per spike
- **Usage**: `DELTA_RP = ETA_RP` (aliased)
- **Both cells**: 10 mV
- **Effect**: Stronger refractoriness after spiking

---

### 📊 **Spike Rate Adaptation (Simple cells only)**

#### **ETA_SRA** (Spike Rate Adaptation)
- **Role**: Adaptation potential increment per spike
- **Usage**: `m_adaptationPotential += ETA_SRA` after each spike
- **Value**: 0 (disabled by default)
- **Effect**: When enabled, reduces excitability after sustained firing

#### **ETA_TA** (Threshold Adaptation)
- **Role**: Rate of threshold adjustment based on firing rate
- **Usage**: Adjusts VTHRESH toward TARGET_SPIKE_RATE
- **Value**: 0 (disabled by default)
- **Effect**: Homeostatic mechanism to maintain target firing rate

#### **TAU_SRA** (Adaptation Time Constant)
- **Role**: Decay rate of adaptation potential
- **Usage**: `m_adaptationPotential *= exp(-Δt / TAU_SRA)`
- **Value**: 100 ms
- **Effect**: How long adaptation effects persist

#### **TARGET_SPIKE_RATE**
- **Role**: Target firing rate for threshold adaptation
- **Value**: 0.75 spikes/second
- **Effect**: Threshold increases/decreases to match this rate

#### **MIN_THRESH**
- **Role**: Minimum allowed threshold during adaptation
- **Value**: 4 mV
- **Effect**: Prevents threshold from becoming too low

---

### 🔧 **Other Parameters**

#### **DECAY_RATE**
- **Role**: Learning rate decay over time
- **Usage**: `m_decay = 1 / (1 + DECAY_RATE * count)`
- **Value**: 0 (disabled)
- **Effect**: When enabled, slows learning over time

#### **SYNAPSE_DELAY** (Simple cells only)
- **Role**: Axonal delay between layers
- **Usage**: Creates delayed copies of events
- **Value**: 0 μs (no delay)
- **Effect**: Can model biological conduction delays

#### **STDP_LEARNING**
- **Role**: Which connections undergo plasticity
- **Options**: "excitatory", "inhibitory", "all", "none"
- **Value**: "all"
- **Effect**: Controls which weight types are updated

#### **POTENTIAL_TRACK**
- **Role**: Coordinates [x, y] of neuron to track for debugging
- **Value**: [4, 4] (simple), [2, 3] (complex)
- **Effect**: Only used for visualization/debugging

#### **TRACKING**
- **Role**: Whether to record spike trains
- **Options**: "none", "partial", "all"
- **Value**: "none"
- **Effect**: Enables/disables spike time recording

---

### 🎯 **Summary: Weight Control Hierarchy**

1. **Individual weight updates** use soft bounds (max ~3)
2. **After STDP**, weights are normalized:
   - Excitatory → `NORM_FACTOR` (50 for simple, 1000 for complex)
   - Local inhibition → `ETA_INH` (1500 for simple, 600 for complex)
   - Lateral inhibition → `LATERAL_NORM_FACTOR` (6500)
   - Topdown → `TOPDOWN_NORM_FACTOR` (2000)
3. **Learning rates** (ETA_LTP, ETA_LTD, ETA_ILTP, ETA_ILTD) control speed
4. **Temporal windows** (TAU_LTP, TAU_LTD) control STDP timing

**To change overall weight strength**: Modify `NORM_FACTOR`, `ETA_INH`, `LATERAL_NORM_FACTOR`, `TOPDOWN_NORM_FACTOR`  
**To change learning speed**: Modify `ETA_LTP`, `ETA_LTD`, `ETA_ILTP`, `ETA_ILTD`  
**To change selectivity**: Modify `VTHRESH`

---

## How to Enable Network Features

All the features you need can be enabled through **configuration changes only** - no C++ source code modifications required!

### 🔧 Configuration File Location
Edit: `networks/test_network/configs/network_config.json`

---

### 1️⃣ Enable Top-Down (Feedback) Connections

**What it does**: Allows higher layers to send predictive feedback to lower layers (key PCL feature)

**Current config**:
```json
"layerInhibitions": [
    ["local"],
    ["local"]
]
```

**Change to**:
```json
"layerInhibitions": [
    ["local"],
    ["local", "topdown"]
]
```

**Explanation**: 
- Add `"topdown"` to the inhibition list for layers that provide feedback to the previous layer
- This enables top-down inhibitory AND excitatory connections
- The C++ code will automatically create and save `tdi.npy` and `tde.npy` files
- Typically applied to layer 0 (simple cells) receiving feedback from layer 1 (complex cells)

**Network config example**:
```json
{
    "layerInhibitions": [
        ["local"],  // Layer 0: receives top-down from layer 1
        ["local", "topdown"]               // Layer 1: only local
                                           //inhibition
    ],
    "interLayerConnections": [
        [-1],  // Layer 0: input layer
        [0]    // Layer 1: connects to layer 0
    ]
}
```

---

### 2️⃣ Enable Lateral (Spatial) Inhibition

**What it does**: Neurons inhibit their spatial neighbors (winner-take-all competition)

**Current config**:
```json
"layerInhibitions": [
    ["local"],
    ["local"]
]
```

**Change to**:
```json
"layerInhibitions": [
    ["local", "lateral"],
    ["local", "lateral"]
]
```

**Additional required config**:
```json
"neuronInhibitionRange": [4, 4]
```

**Explanation**:
- Add `"lateral"` to enable dynamic lateral inhibition between nearby neurons
- `neuronInhibitionRange`: [x_range, y_range] defines the spatial extent
  - `[4, 4]` means each neuron inhibits neighbors within ±4 neurons in x and y
- The C++ code will create and save `li.npy` weight files
- `"local"` inhibition always remains (fast local competition within same spatial location)

**Complete example**:
```json
{
    "layerInhibitions": [
        ["local", "lateral"],
        ["local", "lateral"]
    ],
    "neuronInhibitionRange": [4, 4]  // ±4 neurons in each direction
}
```

**Note on Lateral Excitation**: When you enable `"lateral"` inhibition, the C++ code also automatically creates **lateral excitation** weights (`le.npy` files) alongside inhibition weights. However:
- Lateral excitation is less commonly used in PCL (cooperative vs competitive dynamics)
- The excitation weights are initialized but may have minimal effect depending on neuron parameters
- Use `visualize_excitation_weights()` to visualize these if needed

---

### 3️⃣ Enable Gabor Filters & Orientation Analysis

**What it does**: Analyzes learned receptive fields, fits Gabor functions, computes orientation tuning

**This is POST-TRAINING analysis** - not a training-time configuration!

**Steps**:

1. **Train your network first** (as you did)

2. **Generate orientation responses**: Run stimuli at different orientations and record responses

3. **Use Python analysis tools**:
```python
from src.spiking_network.gabor2 import fit
from src.spiking_network.analysis.network_display import display_network

# Fit Gabor functions to learned weights
gf = fit.GaborFit()
basis = trained_network.generate_weight_mat2()

for idx, weight in enumerate(basis):
    _, params, error = gf.fit(np.expand_dims(weight, axis=2))
    # params contains: sigma, lambda, phase, theta (orientation)
```

4. **Orientation tuning curves**: See notebooks:
   - `RFs and Gabor fitting.ipynb`
   - `Orientation tuned and cross orientation suppressions.ipynb`
   - `tuningCurves.ipynb`

**What gets created**:
- `statistics/orientations/orientations.json` - orientation preferences
- `gabors/0/rotation_response.npy` - responses to rotated stimuli
- Gabor fit parameters for each neuron

**Note**: The `layerCellTypes` key only appears after Gabor analysis is complete. Use `neuronType` instead for configuration.

---

### 🔄 Complete Configuration Example

Here's a full network config with all features enabled:

```json
{
    "interLayerConnections": [[-1], [0]],
    "layerInhibitions": [
        ["local", "lateral"],  // Layer 0: all features
        ["local", "lateral", "topdown"]               // Layer 1: 
                                                  //lateral + local
    ],
    "layerPatches": [
        [[150], [100], [0]],
        [[0], [0], [0]]
    ],
    "layerSizes": [
        [9, 9, 64],   // Layer 0: 9x9 grid, 64 orientations
        [6, 6, 32]    // Layer 1: 6x6 grid, 32 feature types
    ],
    "neuronInhibitionRange": [4, 4],  // Lateral inhibition range
    "nbCameras": 1,
    "neuron1Synapses": 1,
    "neuronOverlap": [
        [3, 3, 1],
        [3, 3, 64]
    ],
    "neuronSizes": [
        [10, 10, 1],
        [4, 4, 64]
    ],
    "neuronType": ["SimpleCell", "ComplexCell"],
    "sharingType": "patch",
    "vfHeight": 260,
    "vfWidth": 346
}
```

---

### 📋 Summary Table

| Feature | Config Key | Values | Files Created |
|---------|-----------|---------|---------------|
| **Top-Down** | `layerInhibitions` | Add `"topdown"` | `Ntdi.npy`, `Ntde.npy` |
| **Lateral Inhibition** | `layerInhibitions` | Add `"lateral"` | `Nli.npy`, `Nle.npy` |
| **Lateral Excitation** | (Automatic with `"lateral"`) | - | `Nle.npy` |
| **Inhibition Range** | `neuronInhibitionRange` | `[x, y]` | - |
| **Gabor Analysis** | Python post-processing | - | `orientations.json`, `rotation_response.npy` |

---

