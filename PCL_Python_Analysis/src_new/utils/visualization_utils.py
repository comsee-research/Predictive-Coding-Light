"""
Visualization utilities for PCL Spiking Neural Networks

This module provides high-level functions to visualize trained networks,
including structure, weights, and connection patterns.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple

# Import from the original src.spiking_network package
from src.spiking_network.network.neuvisys import SpikingNetwork
from src.spiking_network.analysis.network_display import display_network
from src.spiking_network.analysis.network_statistics import (
    visualize_inhibition_weights,
    visualize_excitation_weights,
    visualize_td_inhibition,
    visualize_td_excitation
)


def print_network_structure(network: SpikingNetwork, verbose: bool = True) -> None:
    """
    Display comprehensive network structure and weight statistics.
    
    Args:
        network: Trained SpikingNetwork instance
        verbose: If True, print detailed weight statistics
    """
    if network is None:
        print("⚠ Network not loaded. Train the network first.")
        return
    
    print("=" * 60)
    print("NETWORK STRUCTURE")
    print("=" * 60)
    print(f"\nNetwork path: {network.path}")
    print(f"Number of layers: {len(network.neurons)}")
    print(f"\nLayer shapes (rows x cols x depth):")
    for i, shape in enumerate(network.l_shape):
        print(f"  Layer {i}: {shape[0]} x {shape[1]} x {shape[2]} = {np.prod(shape)} neurons")
    
    print(f"\nTotal neurons in network: {network.nb_neurons}")
    print(f"Neuron types per layer: {network.conf['neuronType']}")
    print(f"Weight sharing type: {network.conf['sharingType']}")
    print(f"Number of cameras: {network.conf['nbCameras']}")
    print(f"Layer inhibition types: {network.conf['layerInhibitions']}")
    
    if verbose:
        # Display weight statistics
        print("\n" + "=" * 60)
        print("WEIGHT STATISTICS")
        print("=" * 60)
        for layer_idx, weights in enumerate(network.weights):
            if len(weights) > 0:
                print(f"\nLayer {layer_idx} ({network.conf['neuronType'][layer_idx]}):")
                print(f"  Weight array shape: {np.array(weights).shape}")
                print(f"  Weight range: [{np.min(weights):.4f}, {np.max(weights):.4f}]")
                print(f"  Weight mean: {np.mean(weights):.4f}, std: {np.std(weights):.4f}")


def generate_weight_pdfs(network: SpikingNetwork, display_pdf: bool = True) -> Optional[str]:
    """
    Generate PDF visualizations of all neuron weights.
    
    Args:
        network: Trained SpikingNetwork instance
        display_pdf: If True, attempt to display the first PDF
        
    Returns:
        Path to the generated PDF file, or None if failed
    """
    if network is None:
        print("⚠ Network not loaded.")
        return None
    
    print("Generating weight images for all neurons...")
    print("This will create visual representations of learned receptive fields.\n")
    
    # Generate weight images - creates PNG files in the images/ directory
    display_network([network])
    
    print(f"✓ Weight visualizations saved to: {network.path}figures/")
    print(f"  - Simple cell weights: {network.path}figures/0/")
    if len(network.neurons) > 1 and len(network.neurons[1]) > 0:
        print(f"  - Complex cell weights: {network.path}figures/1/")
    
    # Determine PDF path based on configuration
    if network.conf["sharingType"] == "patch" and network.conf["nbCameras"] == 2:
        pdf_path = network.path + "figures/0/weight_sharing_combined.pdf"
    elif network.conf["sharingType"] == "patch":
        pdf_path = network.path + "figures/0/weight_sharing_0.pdf"
    else:
        pdf_path = network.path + "figures/0/0_combined.pdf"
    
    if display_pdf and os.path.exists(pdf_path):
        try:
            from pdf2image import convert_from_path
            from IPython.display import display
            print(f"\nDisplaying simple cell weights from: {pdf_path}")
            display(convert_from_path(pdf_path)[0])
        except Exception as e:
            print(f"Note: Could not display PDF: {e}")
            print(f"Weight images are available as PNG files in {network.path}images/0/")
    
    return pdf_path if os.path.exists(pdf_path) else None


def visualize_neuron_receptive_fields(
    network: SpikingNetwork,
    layer_id: int = 0,
    neuron_id: Optional[int] = None,
    max_images: int = 4
) -> None:
    """
    Visualize receptive fields for a specific neuron.
    
    Args:
        network: Trained SpikingNetwork instance
        layer_id: Layer index to visualize
        neuron_id: Neuron index (if None, uses middle neuron)
        max_images: Maximum number of weight images to display
    """
    if network is None or len(network.neurons) == 0:
        print("⚠ Network not loaded or has no neurons.")
        return
    
    print("=" * 60)
    print("INDIVIDUAL NEURON RECEPTIVE FIELDS")
    print("=" * 60)
    
    if len(network.neurons[layer_id]) == 0:
        print(f"⚠ No neurons in layer {layer_id}")
        return
    
    # Pick the middle neuron if not specified
    if neuron_id is None:
        neuron_id = len(network.neurons[layer_id]) // 2
    
    neuron = network.neurons[layer_id][neuron_id]
    
    print(f"\nVisualizing neuron {neuron_id} from layer {layer_id}")
    print(f"Neuron position: {neuron.params['position']}")
    
    # Display the neuron's weight images if they exist
    if hasattr(neuron, 'weight_images') and len(neuron.weight_images) > 0:
        print(f"Number of weight images: {len(neuron.weight_images)}")
        
        try:
            # Create subplots
            num_imgs = min(len(neuron.weight_images), max_images)
            fig, axes_array = plt.subplots(1, num_imgs, figsize=(num_imgs * 3, 3), squeeze=False)
            axes = axes_array[0]  # Get first row
            
            for idx, img_path in enumerate(neuron.weight_images[:num_imgs]):
                if os.path.exists(img_path):
                    img = plt.imread(img_path)
                    axes[idx].imshow(img, cmap='gray')
                    axes[idx].set_title(f"Weight {idx}")
                    axes[idx].axis('off')
            
            fig.suptitle(f"Layer {layer_id}, Neuron {neuron_id} - Learned Receptive Fields")
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Could not create visualization plot: {e}")
            print("\nYou can view the weight images directly at:")
            for idx, img_path in enumerate(neuron.weight_images[:max_images]):
                print(f"  {idx}: {img_path}")
    else:
        print("Weight images not yet generated. Run generate_weight_pdfs() first.")
    
    # Display weight matrix shape and statistics
    if hasattr(neuron, 'weights') and neuron.weights is not None and not isinstance(neuron.weights, int):
        print(f"\nWeight matrix shape: {neuron.weights.shape}")
        print(f"Weight statistics: min={np.min(neuron.weights):.4f}, "
              f"max={np.max(neuron.weights):.4f}, mean={np.mean(neuron.weights):.4f}")
    else:
        print("\nNote: Weight matrix not yet linked to neuron object.")
        print("Weights are available in network.weights[layer_id]")


def visualize_lateral_inhibition(
    network: SpikingNetwork,
    layer_id: int = 0,
    neuron_id: Optional[int] = None
) -> None:
    """
    Visualize lateral inhibition weights for a neuron.
    
    Args:
        network: Trained SpikingNetwork instance
        layer_id: Layer index
        neuron_id: Neuron index (if None, uses middle neuron)
    """
    if network is None or len(network.neurons) == 0:
        print("⚠ Network not loaded.")
        return
    
    print("=" * 60)
    print("LATERAL INHIBITION WEIGHTS")
    print("=" * 60)
    print("Lateral inhibition implements local competition between nearby neurons.\n")
    
    if len(network.neurons[layer_id]) == 0:
        print(f"⚠ No neurons in layer {layer_id}")
        return
    
    # Pick the middle neuron if not specified
    if neuron_id is None:
        neuron_id = len(network.neurons[layer_id]) // 2
    
    # Check if lateral inhibition weight files exist
    li_file = network.path + f"weights/{layer_id}/{neuron_id}li.npy"
    print(li_file)
    
    if os.path.exists(li_file):
        print(f"Visualizing lateral inhibition for neuron {neuron_id} in layer {layer_id}")
        print("Color scale intensity shows inhibition strength to neighboring neurons.")
        print("Black square = the neuron itself\n")
        
        try:
            visualize_inhibition_weights(network, layer_id, neuron_id)
        except Exception as e:
            print(f"Could not visualize inhibition weights: {e}")
    else:
        print(f"⚠ Lateral inhibition was not enabled for this layer.")
        print(f"Expected file not found: {li_file}")
        print(f"\nThis network uses '{network.conf['layerInhibitions'][layer_id]}' inhibition.")
        print("Lateral inhibition visualization requires separate li.npy weight files.")


def visualize_lateral_excitation(
    network: SpikingNetwork,
    layer_id: int = 0,
    neuron_id: Optional[int] = None
) -> None:
    """
    Visualize lateral excitation weights for a neuron.
    
    Args:
        network: Trained SpikingNetwork instance
        layer_id: Layer index
        neuron_id: Neuron index (if None, uses middle neuron)
    """
    if network is None or len(network.neurons) == 0:
        print("⚠ Network not loaded.")
        return
    
    print("=" * 60)
    print("LATERAL EXCITATION WEIGHTS")
    print("=" * 60)
    print("Lateral excitation implements cooperative dynamics between nearby neurons.\n")
    
    if len(network.neurons[layer_id]) == 0:
        print(f"⚠ No neurons in layer {layer_id}")
        return
    
    # Pick the middle neuron if not specified
    if neuron_id is None:
        neuron_id = len(network.neurons[layer_id]) // 2
    
    # Check if lateral excitation weight files exist
    le_file = network.path + f"weights/{layer_id}/{neuron_id}le.npy"
    
    if os.path.exists(le_file):
        print(f"Visualizing lateral excitation for neuron {neuron_id} in layer {layer_id}")
        print("Blue intensity shows excitation strength to neighboring neurons.")
        print("This is less common than lateral inhibition in PCL networks.\n")
        
        try:
            visualize_excitation_weights(network, layer_id, neuron_id)
        except Exception as e:
            print(f"Could not visualize excitation weights: {e}")
    else:
        print(f"⚠ Lateral excitation was not enabled for this layer.")
        print(f"Expected file not found: {le_file}")
        print(f"\nMost PCL networks use lateral inhibition (competition) rather than")
        print("lateral excitation (cooperation). This is the typical configuration.")


def visualize_topdown_connections(
    network: SpikingNetwork,
    layer_id: int = 0,
    neuron_id: Optional[int] = None
) -> None:
    """
    Visualize top-down feedback connections for a neuron.
    
    Args:
        network: Trained SpikingNetwork instance
        layer_id: Layer index (typically 0 for simple cells)
        neuron_id: Neuron index (if None, uses middle neuron)
    """
    if network is None:
        print("⚠ Network not loaded.")
        return
    
    if len(network.neurons) <= 1:
        print("⚠ Network has only one layer. Top-down connections require multiple layers.")
        return
    
    print("=" * 60)
    print("TOP-DOWN FEEDBACK CONNECTIONS")
    print("=" * 60)
    print("Top-down connections provide predictive feedback from higher layers.\n")
    
    if len(network.neurons[layer_id]) == 0:
        print(f"⚠ No neurons in layer {layer_id}")
        return
    
    # Pick the middle neuron if not specified
    if neuron_id is None:
        neuron_id = len(network.neurons[layer_id]) // 2
    
    tdi_file = network.path + f"weights/{layer_id}/{neuron_id}tdi.npy"
    tde_file = network.path + f"weights/{layer_id}/{neuron_id}tde.npy"
    
    print(f"Visualizing top-down connections for neuron {neuron_id} in layer {layer_id}")
    print("This shows how higher-layer neurons modulate this neuron's activity.\n")
    print(f"Checking for top-down weight files...")
    print(f"  - Top-down inhibition: {os.path.exists(tdi_file)}")
    print(f"  - Top-down excitation: {os.path.exists(tde_file)}")
    
    if os.path.exists(tdi_file):
        try:
            print("\nVisualizing top-down inhibition...")
            visualize_td_inhibition(network, layer_id, neuron_id)
            print("Top-down inhibition visualization complete.")
        except Exception as e:
            print(f"Could not visualize top-down inhibition: {e}")
    else:
        print("\n⚠ Top-down inhibition weights not found.")
    
    if os.path.exists(tde_file):
        try:
            print("\nVisualizing top-down excitation...")
            visualize_td_excitation(network, layer_id, neuron_id)
            print("Top-down excitation visualization complete.")
        except Exception as e:
            print(f"Could not visualize top-down excitation: {e}")
    else:
        print("⚠ Top-down excitation weights not found.")
    
    if not (os.path.exists(tdi_file) or os.path.exists(tde_file)):
        print("\n" + "=" * 60)
        print("No top-down connections found.")
        print("Your network config shows:")
        print(f"  - Inter-layer connections: {network.conf['interLayerConnections']}")
        print(f"  - Layer inhibitions: {network.conf['layerInhibitions']}")
        print("\nTo enable top-down connections, add 'topdown' to layerInhibitions")
        print("in your network configuration file.")


def visualize_all_connections(
    network: SpikingNetwork,
    layer_id: int = 0,
    neuron_id: Optional[int] = None,
    include_excitation: bool = False
) -> None:
    """
    Comprehensive visualization of all connection types for a neuron.
    
    Args:
        network: Trained SpikingNetwork instance
        layer_id: Layer index
        neuron_id: Neuron index (if None, uses middle neuron)
        include_excitation: If True, also visualize excitation weights
    """
    if network is None:
        print("⚠ Network not loaded.")
        return
    
    if neuron_id is None and len(network.neurons[layer_id]) > 0:
        neuron_id = len(network.neurons[layer_id]) // 2
    
    print("\n" + "=" * 70)
    print(f"COMPREHENSIVE CONNECTION VISUALIZATION - Layer {layer_id}, Neuron {neuron_id}")
    print("=" * 70 + "\n")
    
    # Lateral inhibition
    visualize_lateral_inhibition(network, layer_id, neuron_id)
    print()
    
    # Lateral excitation (optional)
    if include_excitation:
        visualize_lateral_excitation(network, layer_id, neuron_id)
        print()
    
    # Top-down connections (if multi-layer)
    if len(network.neurons) > 1:
        visualize_topdown_connections(network, layer_id, neuron_id)
