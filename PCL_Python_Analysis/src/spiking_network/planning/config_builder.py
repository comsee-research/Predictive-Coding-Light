#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ConfigBuilder - A fluent API for building PCL network configurations.

This module provides a builder pattern for creating and modifying PCL network
configurations. It supports method chaining for a clean, readable API.

Example usage:
    config = (ConfigBuilder.load("/path/to/network")
              .enable_spike_recording()
              .record_layers([0, 1])
              .set_max_samples_per_class(1000)
              .set_output_subfolder("gesture_spikes")
              .build())
    
    # Or create from scratch with defaults:
    config = (ConfigBuilder.default()
              .set_network_path("/path/to/new_network")
              .set_layer_sizes([[9, 9, 64], [6, 6, 32]])
              .enable_spike_recording()
              .build())

@author: Generated for PCL spike recording generalization
"""

import json
import os
import copy
from typing import List, Dict, Any, Optional, Union
from pathlib import Path


class ConfigBuilder:
    """
    A fluent builder for PCL network configuration.
    
    Supports loading existing configs, modifying them with method chaining,
    and saving the result.
    """
    
    def __init__(self):
        """Initialize empty config containers."""
        self._network_config: Dict[str, Any] = {}
        self._simple_cell_config: Dict[str, Any] = {}
        self._complex_cell_config: Dict[str, Any] = {}
        self._network_path: Optional[str] = None
        self._modified = False  # Track if config has been modified
        
    # ==================== Factory Methods ====================
    
    @classmethod
    def load(cls, network_path: str) -> 'ConfigBuilder':
        """
        Load configuration from an existing network directory.
        
        Args:
            network_path: Path to the network folder (containing configs/ subdirectory)
            
        Returns:
            ConfigBuilder instance with loaded configuration
            
        Raises:
            FileNotFoundError: If config files don't exist
        """
        builder = cls()
        builder._network_path = network_path.rstrip('/')
        config_path = f"{builder._network_path}/configs/"
        
        # Load all config files
        with open(f"{config_path}network_config.json", "r") as f:
            builder._network_config = json.load(f)
        with open(f"{config_path}simple_cell_config.json", "r") as f:
            builder._simple_cell_config = json.load(f)
        with open(f"{config_path}complex_cell_config.json", "r") as f:
            builder._complex_cell_config = json.load(f)
            
        return builder
    
    @classmethod
    def default(cls) -> 'ConfigBuilder':
        """
        Create a ConfigBuilder with default PCL configuration values.
        
        Returns:
            ConfigBuilder instance with sensible defaults
        """
        builder = cls()
        
        # Default network config
        builder._network_config = {
            "nbCameras": 1,
            "neuron1Synapses": 1,
            "sharingType": "patch",
            "vfWidth": 346,
            "vfHeight": 260,
            "measurementInterval": 100,
            "neuronInhibitionRange": [4, 4],
            "neuronType": ["SimpleCell", "ComplexCell"],
            "layerInhibitions": [
                ["local", "lateral"],
                ["local", "topdown"]
            ],
            "interLayerConnections": [[-1], [0]],
            "layerPatches": [[[0], [0], [0]], [[0], [0], [0]]],
            "layerSizes": [[9, 9, 64], [6, 6, 32]],
            "neuronSizes": [[[10, 10, 1]], [[4, 4, 64]]],
            "neuronOverlap": [[3, 3, 0], [3, 3, 64]],
            # Spike recording defaults (disabled)
            "spikeRecording": {
                "enabled": False,
                "layersToRecord": [],
                "maxSamplesPerClass": 0,
                "outputSubfolder": "recorded_spikes"
            }
        }
        
        # Default simple cell config
        # Mapping to table parameters:
        #   V_reset=VRESET, V_thresh=VTHRESH, τ_m=TAU_M, τ_RP=TAU_RP,
        #   τ_LTP=TAU_LTP, τ_LTD=TAU_LTD, η_RP=ETA_RP (refractory delta, stored as DELTA_RP in C++)
        #   λ excit=NORM_FACTOR, λ loc.inhib=ETA_INH, λ dist.inhib=LATERAL_NORM_FACTOR,
        #   λ td.inhib=TOPDOWN_NORM_FACTOR
        #   Excit η_LTP/LTD = ETA_LTP/ETA_LTD
        #   Inhib effective η_LTP/LTD = ETA_ILTP × λ_type (single ETA_ILTP shared across inhib types)
        #   η_+/- (soft-bound factors) and w_max are hardcoded in C++ weightUpdate()
        builder._simple_cell_config = {
            "ETA_LTP": 0.00000408,           # excit η_LTP/LTD = ±0.000204
            "ETA_LTD": -0.00000408,
            "ETA_ILTP": 0.000408,          # inhib learning rate; effective rates = ETA_ILTP × λ:
                                            #   dist: 0.000408 × 6500 = 2.652
                                            #   loc:  0.000408 × 1500 = 0.612
                                            #   td:   0.000408 × 3500 = 1.428
            "ETA_ILTD": -0.000408,
            "ETA_TA": 0,
            "ETA_RP": 10,                  # η_RP (refractory potential delta)
            "ETA_SRA": 0,
            "ETA_INH": 1500,               # λ loc. inhib. (local inhibition norm factor)
            "TAU_LTP": 7,                  # τ_LTP (ms)
            "TAU_LTD": 7,                  # τ_LTD (ms)
            "TAU_M": 18,                   # τ_m (ms)
            "TAU_RP": 5,                   # τ_RP (ms)
            "TAU_SRA": 18,
            "VTHRESH": 10,                 # V_thresh (mV)
            "VRESET": -10,                 # V_reset (mV)
            "SYNAPSE_DELAY": 0,
            "NORM_FACTOR": 50,             # λ excit (excitatory norm factor)
            "LATERAL_NORM_FACTOR": 6500,   # λ dist. inhib. (lateral/distant inhibition norm factor)
            "TOPDOWN_NORM_FACTOR": 3500,   # λ td. inhib. (top-down inhibition norm factor)
            "DECAY_RATE": 0,
            "TARGET_SPIKE_RATE": 0.75,
            "MIN_THRESH": 4,
            "STDP_LEARNING": "all",
            "TRACKING": "none",
            "POTENTIAL_TRACK": [-4, -4],
            "DECAY_LEARNING": 0
        }
        
        # Default complex cell config
        # Same key mapping as simple cells (see comments above)
        # Complex cells only have local inhibition (no lateral/topdown)
        builder._complex_cell_config = {
            "ETA_LTP": 0.00002,            # excit η_LTP/LTD: effective = 0.00008 × 1000 = 0.08
            "ETA_LTD": -0.00002,
            "ETA_ILTP": 0.00008,           # loc inhib η_LTP/LTD: effective = 0.00008 × 600 = 0.048
            "ETA_ILTD": -0.00008,
            "ETA_INH": 600,                # λ loc. inhib. = 600
            "ETA_RP": 10,                   # η_RP = 5
            "TAU_LTP": 40,                 # τ_LTP = 40 ms
            "TAU_LTD": 40,                 # τ_LTD = 40 ms
            "TAU_M": 50,                   # τ_m = 50 ms
            "TAU_RP": 5,                   # τ_RP = 5 ms
            "VTHRESH": 3,                  # V_thresh = 3 mV
            "VRESET": -10,                 # V_reset = -10 mV
            "NORM_FACTOR": 1000,           # λ excit = 1000
            "DECAY_RATE": 0,
            "STDP_LEARNING": "all",
            "TRACKING": "none",
            "POTENTIAL_TRACK": [-2, -3]
        }
        
        return builder
    
    @classmethod
    def from_dict(cls, network_config: Dict, simple_cell_config: Dict, 
                  complex_cell_config: Dict) -> 'ConfigBuilder':
        """
        Create a ConfigBuilder from dictionary configurations.
        
        Args:
            network_config: Network configuration dictionary
            simple_cell_config: Simple cell configuration dictionary
            complex_cell_config: Complex cell configuration dictionary
            
        Returns:
            ConfigBuilder instance
        """
        builder = cls()
        builder._network_config = copy.deepcopy(network_config)
        builder._simple_cell_config = copy.deepcopy(simple_cell_config)
        builder._complex_cell_config = copy.deepcopy(complex_cell_config)
        return builder
    
    # ==================== Spike Recording Configuration ====================
    
    def enable_spike_recording(self, enabled: bool = True) -> 'ConfigBuilder':
        """
        Enable or disable spike recording.
        
        Args:
            enabled: Whether to enable spike recording (default: True)
            
        Returns:
            self for method chaining
        """
        self._ensure_spike_recording_section()
        self._network_config["spikeRecording"]["enabled"] = enabled
        self._modified = True
        return self
    
    def disable_spike_recording(self) -> 'ConfigBuilder':
        """
        Disable spike recording.
        
        Returns:
            self for method chaining
        """
        return self.enable_spike_recording(False)
    
    def record_layers(self, layers: List[int]) -> 'ConfigBuilder':
        """
        Set which layers to record spikes from.
        
        Args:
            layers: List of layer indices to record. Empty list means all layers.
            
        Returns:
            self for method chaining
        """
        self._ensure_spike_recording_section()
        self._network_config["spikeRecording"]["layersToRecord"] = layers
        self._modified = True
        return self
    
    def record_all_layers(self) -> 'ConfigBuilder':
        """
        Record spikes from all layers.
        
        Returns:
            self for method chaining
        """
        return self.record_layers([])
    
    def set_max_samples_per_class(self, max_samples: int) -> 'ConfigBuilder':
        """
        Set maximum number of samples to record per class.
        
        Args:
            max_samples: Maximum samples per class. 0 means unlimited.
            
        Returns:
            self for method chaining
        """
        self._ensure_spike_recording_section()
        self._network_config["spikeRecording"]["maxSamplesPerClass"] = max_samples
        self._modified = True
        return self
    
    def set_output_subfolder(self, subfolder: str) -> 'ConfigBuilder':
        """
        Set the output subfolder name for recorded spikes.
        
        Args:
            subfolder: Name of subfolder within statistics/ directory
            
        Returns:
            self for method chaining
        """
        self._ensure_spike_recording_section()
        self._network_config["spikeRecording"]["outputSubfolder"] = subfolder
        self._modified = True
        return self
    
    # ==================== Network Configuration ====================
    
    def set_network_path(self, path: str) -> 'ConfigBuilder':
        """
        Set the network path for saving.
        
        Args:
            path: Path to network directory
            
        Returns:
            self for method chaining
        """
        self._network_path = path.rstrip('/')
        return self
    
    def set_visual_field(self, width: int, height: int) -> 'ConfigBuilder':
        """
        Set the visual field dimensions.
        
        Args:
            width: Visual field width
            height: Visual field height
            
        Returns:
            self for method chaining
        """
        self._network_config["vfWidth"] = width
        self._network_config["vfHeight"] = height
        self._modified = True
        return self
    
    def set_layer_sizes(self, sizes: List[List[int]]) -> 'ConfigBuilder':
        """
        Set the layer sizes.
        
        Args:
            sizes: List of [width, height, features] for each layer
            
        Returns:
            self for method chaining
        """
        self._network_config["layerSizes"] = sizes
        self._modified = True
        return self
    
    def set_layer_patches(self, patches: List[List[List[int]]]) -> 'ConfigBuilder':
        """
        Set the layer patches.
        
        Args:
            patches: Nested list of patches per layer
            
        Returns:
            self for method chaining
        """
        self._network_config["layerPatches"] = patches
        self._modified = True
        return self
    
    def set_neuron_sizes(self, sizes: List[List[List[int]]]) -> 'ConfigBuilder':
        """
        Set the neuron receptive field sizes.
        
        Args:
            sizes: Nested list of neuron sizes per layer
            
        Returns:
            self for method chaining
        """
        self._network_config["neuronSizes"] = sizes
        self._modified = True
        return self
    
    def set_neuron_overlap(self, overlap: List[List[int]]) -> 'ConfigBuilder':
        """
        Set the neuron overlap for each layer.
        
        Args:
            overlap: List of [x_overlap, y_overlap, z_overlap] per layer
            
        Returns:
            self for method chaining
        """
        self._network_config["neuronOverlap"] = overlap
        self._modified = True
        return self
    
    # ==================== Simple Cell Configuration ====================
    
    def set_simple_cell_param(self, key: str, value: Any) -> 'ConfigBuilder':
        """
        Set a simple cell parameter.
        
        Args:
            key: Parameter name
            value: Parameter value
            
        Returns:
            self for method chaining
        """
        self._simple_cell_config[key] = value
        self._modified = True
        return self
    
    def set_simple_cell_learning_rates(self, eta_ltp: float, eta_ltd: float) -> 'ConfigBuilder':
        """
        Set simple cell STDP learning rates.
        
        Args:
            eta_ltp: LTP learning rate (positive)
            eta_ltd: LTD learning rate (negative)
            
        Returns:
            self for method chaining
        """
        self._simple_cell_config["ETA_LTP"] = eta_ltp
        self._simple_cell_config["ETA_LTD"] = eta_ltd
        self._modified = True
        return self
    
    def set_simple_cell_tracking(self, tracking: str, potential_track: List[int] = None) -> 'ConfigBuilder':
        """
        Set simple cell tracking mode.
        
        Args:
            tracking: "none", "partial", or "all"
            potential_track: [x, y] coordinates to track (optional)
            
        Returns:
            self for method chaining
        """
        self._simple_cell_config["TRACKING"] = tracking
        if potential_track is not None:
            self._simple_cell_config["POTENTIAL_TRACK"] = potential_track
        self._modified = True
        return self
    
    # ==================== Complex Cell Configuration ====================
    
    def set_complex_cell_param(self, key: str, value: Any) -> 'ConfigBuilder':
        """
        Set a complex cell parameter.
        
        Args:
            key: Parameter name
            value: Parameter value
            
        Returns:
            self for method chaining
        """
        self._complex_cell_config[key] = value
        self._modified = True
        return self
    
    def set_complex_cell_learning_rates(self, eta_ltp: float, eta_ltd: float) -> 'ConfigBuilder':
        """
        Set complex cell STDP learning rates.
        
        Args:
            eta_ltp: LTP learning rate (positive)
            eta_ltd: LTD learning rate (negative)
            
        Returns:
            self for method chaining
        """
        self._complex_cell_config["ETA_LTP"] = eta_ltp
        self._complex_cell_config["ETA_LTD"] = eta_ltd
        self._modified = True
        return self
    
    def set_complex_cell_tracking(self, tracking: str, potential_track: List[int] = None) -> 'ConfigBuilder':
        """
        Set complex cell tracking mode.
        
        Args:
            tracking: "none", "partial", or "all"
            potential_track: [x, y] coordinates to track (optional)
            
        Returns:
            self for method chaining
        """
        self._complex_cell_config["TRACKING"] = tracking
        if potential_track is not None:
            self._complex_cell_config["POTENTIAL_TRACK"] = potential_track
        self._modified = True
        return self
    
    # ==================== Generic Parameter Setting ====================
    
    def with_param(self, config_type: str, key: str, value: Any) -> 'ConfigBuilder':
        """
        Generic method to set any parameter in any config file.
        
        Args:
            config_type: "network", "simple_cell", or "complex_cell"
            key: Parameter key
            value: Parameter value
            
        Returns:
            self for method chaining
        """
        if config_type == "network":
            self._network_config[key] = value
        elif config_type == "simple_cell":
            self._simple_cell_config[key] = value
        elif config_type == "complex_cell":
            self._complex_cell_config[key] = value
        else:
            raise ValueError(f"Unknown config type: {config_type}")
        self._modified = True
        return self
    
    # ==================== Build and Save ====================
    
    def build(self) -> Dict[str, Dict[str, Any]]:
        """
        Build and return the complete configuration.
        
        Returns:
            Dictionary with all three config sections
        """
        return {
            "network_config": copy.deepcopy(self._network_config),
            "simple_cell_config": copy.deepcopy(self._simple_cell_config),
            "complex_cell_config": copy.deepcopy(self._complex_cell_config)
        }
    
    def save(self, network_path: Optional[str] = None) -> 'ConfigBuilder':
        """
        Save configuration to disk.
        
        Args:
            network_path: Optional path override. Uses loaded path if not specified.
            
        Returns:
            self for method chaining
            
        Raises:
            ValueError: If no network path is specified
        """
        path = network_path or self._network_path
        if path is None:
            raise ValueError("No network path specified. Use set_network_path() or provide path to save().")
        
        config_path = f"{path}/configs/"
        os.makedirs(config_path, exist_ok=True)
        
        with open(f"{config_path}network_config.json", "w") as f:
            json.dump(self._network_config, f, indent=2)
        with open(f"{config_path}simple_cell_config.json", "w") as f:
            json.dump(self._simple_cell_config, f, indent=2)
        with open(f"{config_path}complex_cell_config.json", "w") as f:
            json.dump(self._complex_cell_config, f, indent=2)
        
        self._modified = False
        return self
    
    # ==================== Utility Methods ====================
    
    def _ensure_spike_recording_section(self):
        """Ensure spikeRecording section exists in network config."""
        if "spikeRecording" not in self._network_config:
            self._network_config["spikeRecording"] = {
                "enabled": False,
                "layersToRecord": [],
                "maxSamplesPerClass": 0,
                "outputSubfolder": "recorded_spikes"
            }
    
    def print_config(self) -> 'ConfigBuilder':
        """
        Print current configuration to stdout.
        
        Returns:
            self for method chaining
        """
        print("=== Network Config ===")
        print(json.dumps(self._network_config, indent=2))
        print("\n=== Simple Cell Config ===")
        print(json.dumps(self._simple_cell_config, indent=2))
        print("\n=== Complex Cell Config ===")
        print(json.dumps(self._complex_cell_config, indent=2))
        return self
    
    def print_spike_recording_config(self) -> 'ConfigBuilder':
        """
        Print only the spike recording configuration.
        
        Returns:
            self for method chaining
        """
        self._ensure_spike_recording_section()
        print("=== Spike Recording Config ===")
        print(json.dumps(self._network_config.get("spikeRecording", {}), indent=2))
        return self
    
    def is_spike_recording_enabled(self) -> bool:
        """Check if spike recording is currently enabled."""
        self._ensure_spike_recording_section()
        return self._network_config["spikeRecording"].get("enabled", False)
    
    def get_layers_to_record(self) -> List[int]:
        """Get list of layers configured for recording."""
        self._ensure_spike_recording_section()
        return self._network_config["spikeRecording"].get("layersToRecord", [])
    
    @property
    def network_config(self) -> Dict[str, Any]:
        """Get the network configuration dictionary."""
        return self._network_config
    
    @property
    def simple_cell_config(self) -> Dict[str, Any]:
        """Get the simple cell configuration dictionary."""
        return self._simple_cell_config
    
    @property
    def complex_cell_config(self) -> Dict[str, Any]:
        """Get the complex cell configuration dictionary."""
        return self._complex_cell_config
    
    @property 
    def is_modified(self) -> bool:
        """Check if configuration has been modified since loading/saving."""
        return self._modified


# ==================== Convenience Functions ====================

def quick_enable_spike_recording(network_path: str, 
                                  layers: Optional[List[int]] = None,
                                  max_samples: int = 0,
                                  output_folder: str = "recorded_spikes") -> None:
    """
    Quickly enable spike recording for an existing network.
    
    Args:
        network_path: Path to network directory
        layers: Layers to record (None = all layers)
        max_samples: Max samples per class (0 = unlimited)
        output_folder: Output subfolder name
    """
    builder = ConfigBuilder.load(network_path)
    builder.enable_spike_recording()
    
    if layers is not None:
        builder.record_layers(layers)
    
    builder.set_max_samples_per_class(max_samples)
    builder.set_output_subfolder(output_folder)
    builder.save()
    
    print(f"✓ Spike recording enabled for {network_path}")
    builder.print_spike_recording_config()


def quick_disable_spike_recording(network_path: str) -> None:
    """
    Quickly disable spike recording for an existing network.
    
    Args:
        network_path: Path to network directory
    """
    (ConfigBuilder.load(network_path)
     .disable_spike_recording()
     .save())
    print(f"✓ Spike recording disabled for {network_path}")
