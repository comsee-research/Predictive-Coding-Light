# PCL Python Analysis Documentation

This directory contains comprehensive documentation for the PCL (Predictive Coding Light) Python analysis tools.

## Available Documentation

### [PCL Network Guide](PCL_Network_Guide.md)
Comprehensive guide to PCL network architecture and configuration:
- Paper terminology vs code implementation mapping
- Detailed explanation of all three inhibition types:
  - Short-ranging lateral (local) inhibition - Winner-Take-All
  - Long-ranging lateral inhibition - Spatial competition
  - Top-down inhibition - Predictive feedback
- Step-by-step configuration instructions
- Complete network configuration examples
- Weight file reference
- Performance considerations

## Quick Links

### Network Configuration
To configure your network, edit: `networks/test_network/configs/network_config.json`

### Visualization Tools
Import from: `src.spiking_network.utils.visualization_utils`

### Example Notebooks
- `Neuvisys_SNN_Training.ipynb` - Main training and visualization notebook
- `RFs and Gabor fitting.ipynb` - Gabor filter analysis
- `Orientation tuned and cross orientation suppressions.ipynb` - Orientation tuning
- `Surround suppression.ipynb` - Spatial surround effects

## Getting Started

1. Review the [PCL Network Guide](PCL_Network_Guide.md) to understand the architecture
2. Run the training notebook: `Neuvisys_SNN_Training.ipynb`
3. Use the visualization utilities to explore your trained network
4. Modify the network configuration to enable different features

## Support

For issues or questions, refer to:
- The main PCL repository README
- Example notebooks in the `PCL_Python_Analysis` directory
- C++ source code in `libs/network/src/`
