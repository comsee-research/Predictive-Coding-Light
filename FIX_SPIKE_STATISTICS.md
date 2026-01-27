# Fix for Spike Statistics Not Being Saved

## Problem

Even with `TRACKING = "partial"` enabled in the neuron configuration files, the statistics directory was not being created. The error was:

```
Statistics directory not found: /path/to/network/statistics/0/0/1
```

## Root Cause

The `neuvisys` executable in [libs/apps/neuvisys/src/main.cpp](libs/apps/neuvisys/src/main.cpp) was **not calling `saveStatistics()`** after processing events. 

The original code only saved network weights:

```cpp
network.save(eventsPath, nbCount);  // Only saves weights, not statistics
```

But to save spike trains (even with TRACKING enabled), the code must explicitly call:

```cpp
network.saveStatistics(simulation_id, sequence_id, "folder_name/");
```

## Solution Applied

### 1. Modified `libs/apps/neuvisys/src/main.cpp`

Added the following line after `network.save()`:

```cpp
// Save statistics with spike trains (simulation=0, sequence=1)
network.saveStatistics(0, 1, "");
```

**Full modified section:**
```cpp
while (network.loadEvents(events, nbCount)) {
    network.feedEvents(events);
}
network.save(eventsPath, nbCount);

// Save statistics with spike trains (simulation=0, sequence=1)
network.saveStatistics(0, 1, "");
```

This ensures that after processing each event file, the spike trains are saved to:
```
{network_path}/statistics/{layer}/{simulation}/{sequence}/{neuron_id}.json
```

Where:
- `layer` = 0 (simple cells) or 1 (complex cells)
- `simulation` = 0 (hardcoded)
- `sequence` = 1 (hardcoded)
- `neuron_id` = 0, 1, 2, ... (one JSON file per neuron)

### 2. Executable Must Be Rebuilt

**Critical:** After modifying `main.cpp`, the executable must be recompiled:

```bash
cd /users/micas/pgraindo/Documents/LifeLines/PCL/Predictive-Coding-Light/build
cmake --build . --target neuvisys-exe
```

Or from Python (added to notebook):
```python
subprocess.run(["cmake", "--build", ".", "--target", "neuvisys-exe"], cwd=build_dir)
```

### 3. Updated Notebook

The notebook [DVS_Gesture128_testing.ipynb](PCL_Python_Analysis/notebooks_new/DVS_Gesture128_testing.ipynb) now includes:

- **Warning section** explaining the rebuild requirement
- **Build cell** to rebuild the executable directly from the notebook
- **Enhanced error checking** to verify statistics are created
- **Debug output** showing when statistics directory is created
- **Reduced initial sample count** (5 samples) to quickly verify the fix works

## Verification Steps

1. **Enable TRACKING** in configs (already done by notebook):
   - `simple_cell_config.json`: `"TRACKING": "partial"`
   - `complex_cell_config.json`: `"TRACKING": "partial"`

2. **Rebuild executable**:
   ```bash
   cd build
   cmake --build . --target neuvisys-exe
   ```

3. **Run a test sample**:
   ```python
   launch_neuvisys_multi_pass(exec_path, network_path, event_file, nb_pass=1)
   ```

4. **Check statistics directory**:
   ```bash
   ls -la {network_path}/statistics/0/0/1/
   ```
   Should show JSON files: `0.json`, `1.json`, `2.json`, etc.

5. **Verify spike_train in JSON**:
   ```bash
   cat {network_path}/statistics/0/0/1/0.json
   ```
   Should contain:
   ```json
   {
       "spike_train": [12500, 25300, 37150, ...],
       "count_spike": 127,
       ...
   }
   ```

## How It Works

1. **TRACKING = "partial"**: Neurons record spike timestamps in `m_trackingSpikeTrain` vector
2. **feedEvents()**: Network processes events, neurons fire and record spikes
3. **saveStatistics()**: Called after all events processed
   - Creates directory structure: `statistics/{layer}/{sim}/{seq}/`
   - Calls `writeJson()` for each neuron
   - Writes `spike_train` field containing timestamp array
4. **Python extraction**: Reads JSON files and counts spikes per neuron as features

## Files Modified

- ✏️ [libs/apps/neuvisys/src/main.cpp](libs/apps/neuvisys/src/main.cpp) - Added `saveStatistics()` call
- 📓 [PCL_Python_Analysis/notebooks_new/DVS_Gesture128_testing.ipynb](PCL_Python_Analysis/notebooks_new/DVS_Gesture128_testing.ipynb) - Added rebuild instructions and verification

## Expected Output Format

After processing one sample, the directory structure should be:

```
network/
└── statistics/
    ├── 0/           # Layer 0 (simple cells)
    │   └── 0/       # Simulation 0
    │       └── 1/   # Sequence 1
    │           ├── 0.json
    │           ├── 1.json
    │           ├── 2.json
    │           └── ... (5184 files for 9x9x64 neurons)
    └── 1/           # Layer 1 (complex cells)
        └── 0/       # Simulation 0
            └── 1/   # Sequence 1
                ├── 0.json
                ├── 1.json
                └── ... (1152 files for 6x6x32 neurons)
```

Each JSON file contains spike train data for one neuron.

## Troubleshooting

### Issue: Statistics directory still not created
- **Check**: Did you rebuild the executable?
- **Verify**: `ls -la build/libs/apps/neuvisys/neuvisys-exe` shows recent timestamp
- **Test**: Run executable manually to see console output

### Issue: Empty spike trains
- **Check**: Is TRACKING set to "partial" in both config files?
- **Verify**: Load config and print TRACKING value
- **Note**: Some neurons may naturally have zero spikes for certain stimuli

### Issue: Build fails
- **Check**: CMake configuration is valid
- **Try**: Clean build: `rm -rf build && mkdir build && cd build && cmake ..`
- **Check**: All dependencies installed (OpenCV, HDF5, etc.)

## Summary

The fix is simple but **requires rebuilding the executable**:
1. Added one line to `main.cpp`: `network.saveStatistics(0, 1, "");`
2. Rebuild: `cmake --build . --target neuvisys-exe`
3. Run notebook - statistics will now be saved!
