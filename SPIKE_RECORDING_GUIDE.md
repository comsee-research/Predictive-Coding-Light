# Spike Recording for DVS-Gesture128 Classification

This guide explains how to use the newly implemented spike recording functionality to extract output spikes from simple and complex cells for DVS-Gesture128 classification.

## Overview

The spike recording implementation allows you to:
1. Feed gesture event sequences through a trained PCL network
2. Record the output spikes from simple and complex cells
3. Save the spike data in JSON format for downstream classification (e.g., SVM training)

## Implementation Details

### Modified Files

1. **`libs/apps/neuvisys/src/main.cpp`**: Added spike recording code after network training
2. **`libs/network/src/SpikingNetwork.cpp`**: Changed output directory from "mnist" to "gesture"

### How It Works

The implementation uses the `classificationDescriptor` function in the `SurroundSuppression` class, which:
- Iterates through all gesture folders
- Processes event files for each gesture class
- Records spike activity for each sample
- Saves output as JSON files organized by gesture label

## Usage Instructions

### Important: Understanding the Two Modes

The neuvisys executable has **two distinct modes**:

1. **Normal Mode** (`recordSpikes=false` or 4th argument omitted):
   - Processes a single event file
   - Used for training or single-sample inference
   - Command: `./neuvisys <network> <event_file> <nb_passes>`

2. **Spike Recording Mode** (`recordSpikes=true`):
   - Processes entire dataset folder structure
   - Iterates through all gesture class folders automatically
   - Records spikes for classification
   - Command: `./neuvisys <network> <gesture_base_folder> 1 true`

### Step 1: Prepare Your Gesture Dataset

Organize your DVS-Gesture128 dataset with numbered prefixes:
```
/path/to/DVS-Gesture128_preprocessed/npz/train/
├── 0_hand_clapping/
│   ├── sample_0001.npz
│   ├── sample_0002.npz
│   └── ...
├── 1_right_hand_wave/
│   ├── sample_0001.npz
│   └── ...
├── 2_left_hand_wave/
...
└── 10_other_gestures/
```

**Note**: The folder names must match those hardcoded in main.cpp (with numeric prefixes).

### Step 2: ~~Enable Spike Recording in main.cpp~~ (Already Configured)

The spike recording functionality is already implemented in main.cpp. It's controlled by the 4th command-line argument.

**No manual code changes needed!** The gesture paths are pre-configured for the folder structure above.

### Step 3: Rebuild the Project

After modifying `main.cpp`, rebuild the project:

```bash
cd /path/to/Predictive-Coding-Light/build
cmake ..
make
```

### Step 4: Run Spike Recording

Execute the neuvisys application with your trained network:

```bash
./build/bin/neuvisys <network_path> <gesture_base_path> 1 true
```

**Important Parameters**:
- `<network_path>`: Path to your trained PCL network (e.g., `./networks/my_network/`)
- `<gesture_base_path>`: **Base path to folder containing gesture subfolders** (e.g., `./data/events/DVS-Gesture128_preprocessed/npz/train/`)
  - Must end with `/` or will be added automatically
  - Should contain subdirectories: `0_hand_clapping/`, `1_right_hand_wave/`, etc.
- `1`: Number of passes (typically 1 for spike recording)
- `true`: Enable spike recording mode

Example:
```bash
./build/bin/neuvisys \
    ./networks/dvsgesture_finetuned_network/ \
    ./data/events/DVS-Gesture128_preprocessed/npz/train/ \
    1 \
    true
```

**How it works**:
- When `recordSpikes=true`, the program skips normal event loading
- Instead, it calls `classificationDescriptor()` which:
  - Automatically appends gesture folder names (`0_hand_clapping`, etc.) to the base path
  - Iterates through ALL .npz files in each gesture folder
  - Processes each file through the network
  - Records spikes to `<network_path>/statistics/gesture/<label>/<sample>.json`

### Step 5: Access the Recorded Spikes

After execution, spike data will be saved in:
```
<network_path>/statistics/gesture/<label_id>/<sample_id>.json
```

For example:
```
./networks/my_network/statistics/gesture/
├── 0/
│   ├── 0.json
│   ├── 1.json
│   └── ...
├── 1/
│   ├── 0.json
│   └── ...
...
└── 10/
```

### Step 6: Load Spikes in Python

You can load the recorded spikes in your Jupyter notebooks:

```python
import json
import numpy as np
from pathlib import Path

network_path = "./networks/my_network/"
spike_dir = Path(network_path) / "statistics" / "gesture"

# Load spikes for gesture class 0
gesture_0_spikes = []
for json_file in sorted((spike_dir / "0").glob("*.json")):
    with open(json_file, 'r') as f:
        data = json.load(f)
        gesture_0_spikes.append({
            'spikes': np.array(data['x']),  # Spike descriptor
            'label': data['y'][0]           # Gesture label
        })

print(f"Loaded {len(gesture_0_spikes)} samples for gesture 0")
```

## Output Format

Each JSON file contains:
- **`x`**: 2D array of spike descriptors (features extracted from neuron activity)
- **`y`**: Array containing the gesture label (0-10)

The spike descriptors capture the temporal activity patterns of simple and complex cells and can be used directly for SVM training.

## Important Parameters in classificationDescriptor

Located in `libs/network/src/SurroundSuppression.cpp` (line 1181):

- **`num_labels`**: Set to 11 for DVS-Gesture128
- **`n_sim`**: Maximum number of samples per gesture (default: 3500)
- **`sigma`**: Time window parameter (default: 5000)

You can modify these parameters if needed before recompiling.

## Troubleshooting

### Issue: "Cannot save neuron state supervised dataset file"
- **Cause**: Insufficient permissions or invalid network path
- **Solution**: Ensure the network path exists and has write permissions

### Issue: No JSON files generated
- **Cause**: Event files not found or incorrect paths
- **Solution**: Verify gesture folder paths are absolute and contain `.h5` event files

### Issue: Compilation errors after modifications
- **Cause**: Syntax errors in modified code
- **Solution**: Ensure you properly uncommented the block and closed all brackets/strings

## Alternative: Using Debug Mode

The original debug mode in `main.cpp` (lines 59-70) demonstrates spike recording with the `recordSpikes` function. This is simpler but doesn't provide label information:

```cpp
// In debug mode (argc <= 1)
SurroundSuppression surround(networkPath, vectorOfPaths, network);
surround.recordSpikes(path_Events);
```

This approach is useful for quick testing but doesn't organize spikes by gesture class.

## Next Steps: SVM Training

Once you have recorded spikes, you can train an SVM classifier:

1. Load all spike descriptors and labels from the JSON files
2. Split into training and testing sets
3. Train a linear SVM on the spike features
4. Evaluate classification accuracy

See the notebooks in `PCL_Python_Analysis/notebooks/` for examples of SVM training on spike data.

## Notes

- The spike recording process processes each gesture sample once
- Processing time depends on the number of samples and network complexity
- Ensure sufficient disk space for the output JSON files (can be large for many samples)
- The function normalizes weights (L1 normalization) before spike recording
