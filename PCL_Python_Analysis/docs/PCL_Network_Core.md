# PCL Network Architecture Guide

This document provides detailed explanations of the Predictive Coding Light (PCL) network architecture, inhibition types, and configuration options.

---

## Inhibition: Paper Terminology vs Code Implementation

The paper describes **three types of inhibitory connections**. Here's how they map to the code:


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

## 🔗 Summary Mapping Table

| Paper Term | Config Value | Code Function | Connection Pattern | Weight Files |
|------------|--------------|---------------|-------------------|--------------|
| **Short-Ranging Lateral** | `"local"` | `lateralLocalInhibitionConnection()` | Same (x,y), different z | `Nlli.npy` |
| **Long-Ranging Lateral** | `"lateral"` | `lateralDynamicInhibitionConnection()` | Within spatial range, all z | `Nli.npy`, (`Nle.npy`) |
| **Top-Down** | `"topdown"` | `topDownConnection()` | Layer N+1 → Layer N (receptive field) | `Ntdi.npy`, (`Ntde.npy`) |

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

### **Weight Control Hierarchy**

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

## How to Enable or Modify Network Features

All the features you need can be enabled through **configuration changes only** - no C++ source code modifications required!

### 🔧 Edit Configuration
- Edit: `networks/your_network/configs/network_config.json`, `networks/your_network/configs/simple_cell_config.json`, and `networks/your_network/configs/complex_cell_config.json`

**OR**

- Use the Python Config API (recommended, see **[PCL_PYTHON_API.md](PCL_PYTHON_API.md)**)  

## 🔄 Complete Configuration Example

Here's a full network config with all PCL features enabled:

```json
{
    "interLayerConnections": [[-1], [0]],
    "layerInhibitions": [
        ["local", "lateral"],              // Layer 0: local + lateral + receives topdown from next layer
        ["local", "lateral", "topdown"]    // Layer 1: lateral + local + send topdown to previous layer
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