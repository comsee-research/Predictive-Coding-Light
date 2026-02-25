# PCL Network Parameters Reference

Complete reference for all PCL network configuration parameters.

## Network Configuration

File: `configs/network_config.json`

### Architecture Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `nbCameras` | int | Number of camera inputs | `1` |
| `vfWidth` | int | Visual field width (pixels) | `66`, `346` |
| `vfHeight` | int | Visual field height (pixels) | `66`, `260` |
| `layerSizes` | array | Layer dimensions `[[w, h, features], ...]` | `[[9,9,64], [6,6,32]]` |
| `neuronSizes` | array | Receptive field sizes `[[[w, h, depth]], ...]` | `[[[10,10,1]], [[4,4,64]]]` |
| `neuronOverlap` | array | Neuron overlap `[[w, h, depth], ...]` | `[[3,3,0], [3,3,0]]` |
| `layerPatches` | array | Input cropping/offset `[[[x], [y], [t]], ...]` | `[[[150],[100],[0]], [[0],[0],[0]]]` |
| `neuronType` | array | Neuron type per layer | `["SimpleNeuron", "ComplexNeuron"]` |
| `sharingType` | string | Weight sharing strategy | `"patch"`, `"none"` |

**layerPatches Explanation:**
- Used to crop or offset the input at each layer
- Format: `[[[x_offset], [y_offset], [time_offset]], ...]` for each layer
- Example: `[[[150], [100], [0]], [[0], [0], [0]]]` crops 346×260 input to view 66×66 region starting at (150,100)

### Inhibition Configuration

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `layerInhibitions` | array | Inhibition types per layer | `[["local","lateral"], ["local","lateral","topdown"]]` |
| `neuronInhibitionRange` | array | Lateral inhibition range `[x, y]` | `[4, 4]` |

**Inhibition Types:**
- `"local"`: Winner-take-all at same spatial location (different feature depths)
- `"lateral"`: Spatial competition within neighborhood (defined by `neuronInhibitionRange`)
- `"topdown"`: Predictive feedback from higher to lower layers

**Example Configurations:**

Full PCL (all inhibition types):
```json
"layerInhibitions": [
    ["local", "lateral"],
    ["local", "lateral", "topdown"]
]
```

Simple hierarchical (no top-down):
```json
"layerInhibitions": [
    ["local", "lateral"],
    ["local", "lateral"]
]
```

### Spike Recording Configuration

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `spikeRecording.enabled` | bool | Enable spike recording | `true` |
| `spikeRecording.layersToRecord` | array | Layers to record (empty = all) | `[0, 1]` |
| `spikeRecording.maxSamplesPerClass` | int | Max samples per class (0 = unlimited) | `1000` |
| `spikeRecording.outputSubfolder` | string | Output folder name | `"experiment_01"` |

**Example:**
```json
"spikeRecording": {
    "enabled": true,
    "layersToRecord": [0, 1],
    "maxSamplesPerClass": 0,
    "outputSubfolder": "gesture_spikes"
}
```

---

## Simple Cell Configuration

File: `configs/simple_cell_config.json`

### Learning Parameters

| Parameter | Type | Description | Typical Value |
|-----------|------|-------------|---------------|
| `ETA_LTP` | float | Long-term potentiation rate (excitatory strengthening) | `0.0003` |
| `ETA_LTD` | float | Long-term depression rate (excitatory weakening) | `-0.0003` |
| `ETA_ILTP` | float | Inhibitory LTP rate | `0.002` |
| `ETA_ILTD` | float | Inhibitory LTD rate | `-0.001` |
| `STDP_LEARNING` | string | Learning mode | `"both"`, `"inhibitory"`, `"none"` |

**STDP_LEARNING modes:**
- `"both"`: Learn both excitatory and inhibitory connections
- `"inhibitory"`: Freeze excitatory weights, learn only inhibitory (fine-tuning)
- `"none"`: Freeze all weights (inference only)

### Membrane Dynamics

| Parameter | Type | Description | Typical Value | Unit |
|-----------|------|-------------|---------------|------|
| `TAU_M` | int | Membrane time constant | `20000` | µs |
| `VTHRESH` | float | Spike threshold voltage | `10.0` | - |
| `VRESET` | float | Reset voltage after spike | `0.0` | - |
| `VRESTING` | float | Resting membrane potential | `0.0` | - |
| `VINIT` | float | Initial membrane potential | `0.0` | - |

### Synaptic Parameters

| Parameter | Type | Description | Typical Value | Unit |
|-----------|------|-------------|---------------|------|
| `TAU_THETA` | int | Threshold adaptation time constant | `100000000` | µs |
| `THETA_PLUS_E` | float | Threshold increase after spike | `0.05` | - |
| `SYNAPSE_DELAY` | int | Synaptic transmission delay | `0` | µs |
| `A_MINUS` | float | STDP depression window amplitude | `1.0` | - |
| `A_PLUS` | float | STDP potentiation window amplitude | `1.0` | - |
| `TAU_MINUS` | int | STDP depression time window | `20000` | µs |
| `TAU_PLUS` | int | STDP potentiation time window | `20000` | µs |


---

## Complex Cell Configuration

File: `configs/complex_cell_config.json`

### Learning Parameters

| Parameter | Type | Description | Typical Value |
|-----------|------|-------------|---------------|
| `ETA_LTP` | float | Long-term potentiation rate | `0.0003` |
| `ETA_LTD` | float | Long-term depression rate | `-0.0003` |
| `ETA_ILTP` | float | Inhibitory LTP rate | `0.002` |
| `ETA_ILTD` | float | Inhibitory LTD rate | `-0.001` |
| `STDP_LEARNING` | string | Learning mode | `"both"`, `"inhibitory"`, `"none"` |

### Membrane Dynamics

| Parameter | Type | Description | Typical Value | Unit |
|-----------|------|-------------|---------------|------|
| `TAU_M` | int | Membrane time constant | `20000` | µs |
| `VTHRESH` | float | Spike threshold voltage | `12.0` | - |
| `VRESET` | float | Reset voltage | `0.0` | - |
| `VRESTING` | float | Resting potential | `0.0` | - |
| `VINIT` | float | Initial potential | `0.0` | - |

### Complex Cell Specific

| Parameter | Type | Description | Typical Value | Unit |
|-----------|------|-------------|---------------|------|
| `SIGMA` | int | Pooling time window | `5000` | µs |
| `TAU_THETA` | int | Threshold adaptation | `100000000` | µs |
| `THETA_PLUS_E` | float | Threshold increase | `0.05` | - |

### Synaptic & Recording

| Parameter | Type | Description | Typical Value |
|-----------|------|-------------|---------------|
| `A_MINUS` | float | STDP depression amplitude | `1.0` |
| `A_PLUS` | float | STDP potentiation amplitude | `1.0` |
| `TAU_MINUS` | int | STDP depression window | `20000` µs |
| `TAU_PLUS` | int | STDP potentiation window | `20000` µs |
| `TRACKING` | string | ??? | `"none"`, `"partial"` |
| `POTENTIAL_TRACK` | array | Neuron to track `[x, y]` | `[0, 0]` |

---

## Parameter Units

- **Time**: Microseconds (µs)
- **Voltage**: Arbitrary units (relative scale)
- **Learning rates**: Dimensionless (typically 0.0001 - 0.001)
- **Spatial**: Pixels or neuron indices

## Common Parameter Sets

### DVS-Gesture Fine-tuning (Inhibitory Only)

```json
// Both simple and complex:
{
    "STDP_LEARNING": "inhibitory",
    // Other params unchanged
}
```

### Inference (No Learning)

```json
// Both simple and complex:
{
    "STDP_LEARNING": "none",
}
```

---

## Architecture Examples

### Standard 2-Layer Network (66×66 input)

```json
{
    "vfWidth": 66,
    "vfHeight": 66,
    "neuronSizes": [[[10, 10, 1]], [[4, 4, 64]]],
    "layerSizes": [[9, 9, 64], [6, 6, 32]],
    "neuronOverlap": [[3, 3, 0], [3, 3, 0]],
    "layerPatches": [[[0], [0], [0]], [[0], [0], [0]]],
    "layerInhibitions": [
        ["local", "lateral"],
        ["local", "lateral", "topdown"]
    ],
    "neuronInhibitionRange": [4, 4]
}
```

### Native DVS Resolution (346×260 → 66×66 crop)

```json
{
    "vfWidth": 346,
    "vfHeight": 260,
    "layerPatches": [[[150], [100], [0]], [[0], [0], [0]]],
    // Rest same as above
}
```
