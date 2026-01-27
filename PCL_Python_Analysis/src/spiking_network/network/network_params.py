def base_learning_params(params=None):
    if params is None:
        params = {}
    param_dict = {
        "network_config": {
            'V0': [0],
            'actionRate': [500],
            'decayRate': [0.02],
            'explorationFactor': [50],
            'interLayerConnections': [[0, 0]],
            'layerCellTypes': [['SimpleCell', 'ComplexCell']],
            'layerInhibitions': [[True, True]],
            'layerPatches': [[[[0], [0], [0]], [[0], [0], [0]]]],
            'layerSizes': [[[16, 16, 64], [4, 4, 16]]],
            'minActionRate': [100],
            'nbCameras': [1],
            'neuron1Synapses': [1],
            'neuronOverlap': [[[0, 0, 0], [0, 0, 0]]],
            'neuronSizes': [[[10, 10, 1], [4, 4, 64]]],
            'nu': [1],
            'saveData': [True],
            'sharingType': ['patch'],
            'tauR': [1]
        },
        "simple_cell_config": {
            'ETA_INH': 20,
            'ETA_LTD': -0.0021,
            'ETA_LTP': 0.0077,
            'ETA_ILTP': 7.7,
            'ETA_ILTD': -2.1,
            'ETA_RP': 1,
            'ETA_SRA': 0.6,
            'ETA_TA': 0,
            'MIN_THRESH': 4,
            'NORM_FACTOR': 4,
            'STDP_LEARNING': "excitatory",
            'SYNAPSE_DELAY': 0,
            'DECAY_LEARNING': 0,
            'TARGET_SPIKE_RATE': 0.75,
            'TAU_LTD': 14,
            'TAU_LTP': 7,
            'TAU_M': 18,
            'TAU_RP': 20,
            'TAU_SRA': 100,
            'TRACKING': 'partial',
            'VRESET': -20,
            'VTHRESH': 4
        },
        "complex_cell_config": {
            'ETA_INH': 15,
            'ETA_LTD': -0.2,
            'ETA_LTP': 0.2,
            'ETA_RP': 1,
            'NORM_FACTOR': 10,
            'STDP_LEARNING': "excitatory",
            'DECAY_LEARNING': 0,
            'TAU_LTD': 20,
            'TAU_LTP': 20,
            'TAU_M': 20,
            'TAU_RP': 20,
            'TRACKING': 'partial',
            'VRESET': -20,
            'VTHRESH': 3
        },
    }
    for key, value in params.items():
        for key2, value2 in value.items():
            param_dict[key][key2] = [value2]
    return param_dict


def reinforcement_learning():
    return base_learning_params(
        {'network_config': {'layerCellTypes': ["SimpleCell", "ComplexCell", "CriticCell", "ActorCell"],
                            'layerInhibitions': [True, True, False, False],
                            'interLayerConnections': [0, 0, 1, 1],
                            'layerPatches': [[[33], [110], [0]], [[0], [0], [0]], [[0], [0], [0]], [[0], [0], [0]]],
                            'layerSizes': [[28, 4, 64], [7, 1, 16], [100, 1, 1], [2, 1, 1]],
                            'neuronSizes': [[10, 10, 1], [4, 4, 64], [13, 1, 16], [13, 1, 16]],
                            'neuronOverlap': [[0, 0, 0], [2, 2, 0], [0, 0, 0], [0, 0, 0]]}
         })


def reinforcement_learning_rotation():
    return base_learning_params(
        {'network_config': {'layerCellTypes': ["SimpleCell", "ComplexCell", "CriticCell", "ActorCell"],
                            'layerInhibitions': [True, True, False, False],
                            'interLayerConnections': [0, 0, 1, 1],
                            'layerPatches': [[[33], [110], [0]], [[0], [0], [0]], [[0], [0], [0]], [[0], [0], [0]]],
                            'layerSizes': [[16, 16, 144], [4, 4, 16], [100, 1, 1], [2, 1, 1]],
                            'neuronSizes': [[10, 10, 1], [4, 4, 144], [4, 4, 16], [4, 4, 16]],
                            'neuronOverlap': [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]},
         'simple_cell_config': {'ETA_LTP': 0.00077,
                                'ETA_LTD': -0.00021}
         })


def inhibition_orientation():
    return base_learning_params(
        {'network_config': {'layerPatches': [[[93], [50], [0]], [[0], [0], [0]]],
                            'layerSizes': [[16, 16, 144], [4, 4, 16]],
                            'neuronSizes': [[10, 10, 1], [4, 4, 144]]},
         'simple_cell_config': {'ETA_LTP': 0.00077,
                                'ETA_LTD': -0.00021}
         })


def inhibition_orientation_9regions():
    return base_learning_params(
        {'network_config': {'layerPatches': [[[0, 153, 306], [0, 110, 220], [0]],
                                             [[0, 4, 8], [0, 4, 8], [0]]],
                            'layerSizes': [[4, 4, 100], [1, 1, 16]],
                            'neuronSizes': [[10, 10, 1], [4, 4, 100]]},
         'simple_cell_config': {'VTHRESH': 25,
                                'ETA_INH': 15,
                                'ETA_LTP': 0.00077,
                                'ETA_LTD': -0.00021}
         })


def inhibition_disparity_9regions():
    return base_learning_params(
        {'network_config': {'nbCameras': 2,
                            'layerPatches': [[[0, 153, 306], [0, 110, 220], [0]],
                                             [[0, 4, 8], [0, 4, 8], [0]]],
                            'layerSizes': [[4, 4, 100], [1, 1, 16]],
                            'neuronSizes': [[10, 10, 1], [4, 4, 100]]},
         'simple_cell_config': {'VTHRESH': 25,
                                'ETA_INH': 15,
                                'ETA_LTP': 0.000077,
                                'ETA_LTD': -0.000021}
         })

def inhibition_disparity():
    return base_learning_params(
        {'network_config': {'nbCameras': 2,
                            'layerPatches': [[[20], [20], [0]], [[0], [0], [0]]],
                            'layerSizes': [[16, 16, 144], [4, 4, 16]],
                            'neuronSizes': [[10, 10, 1], [4, 4, 144]]},
         'simple_cell_config': {'VTHRESH': 25,
                                'ETA_INH': 15,
                                'ETA_LTP': 0.000077,
                                'ETA_LTD': -0.000021}
         })


def natural_learning_params():
    """
    Parameters from Table 2 of the Predictive Coding Light paper for training on natural images.
    
    This function returns only the essential parameters needed for natural images training,
    without RL-specific parameters from base_learning_params.
    
    Parameter mapping from paper to code:
    
    Simple cells (Table 2 - left half):
    - V_reset = -10 mV, V_thresh = 10 mV
    - τ_m = 18 ms, τ_RP = 5 ms, τ_LTP = 7 ms, τ_LTD = 7 ms
    - η_RP = 10 mV (ETA_RP)
    - Excitatory: η_LTP/LTD = ±0.000204 mV, λ = 50 (NORM_FACTOR)
    - Distant inhibition: η_LTP/LTD = ±2.652 mV, η_r = 0.33, λ = 6500 (LATERAL_NORM_FACTOR)
    - Local inhibition: η_LTP/LTD = ±0.816 mV, η_r = 0.02, λ = 1500 (ETA_INH)
    - Top-down inhibition: η_LTP/LTD = ±1.428 mV, η_r = 0.02, λ = 3500 (TOPDOWN_NORM_FACTOR)
    
    Complex cells (Table 2 - right half):
    - V_reset = -10 mV, V_thresh = 3 mV
    - τ_m = 50 ms, τ_RP = 5 ms, τ_LTP = 40 ms, τ_LTD = 40 ms
    - η_RP = 5 mV (ETA_RP)
    - Excitatory: η_LTP/LTD = ±0.08 mV, λ = 1000 (NORM_FACTOR)
    - Local inhibition: η_LTP/LTD = ±0.048 mV, η_r = 0.04, λ = 600 (ETA_INH)
    
    Note: η_r is the ratio parameter that maps to ETA_ILTP/ETA_ILTD in the code.
    """
    return {
        "simple_cell_config": {
            # Membrane dynamics (Table 2)
            'VRESET': -10,              # V_reset (mV)
            'VTHRESH': 10,              # V_thresh (mV)
            'TAU_M': 18,                # τ_m (ms) - membrane time constant
            'TAU_RP': 5,                # τ_RP (ms) - refractory period time constant
            'TAU_LTP': 7,               # τ_LTP (ms) - LTP time constant
            'TAU_LTD': 7,               # τ_LTD (ms) - LTD time constant
            'TAU_SRA': 100,             # Spike rate adaptation (not in table, using default)
            
            # Learning rates (Table 2)
            'ETA_RP': 10,               # η_RP (mV) - refractory period learning rate
            'ETA_LTP': 0.000204,        # η_LTP (mV) - excitatory LTP
            'ETA_LTD': -0.000204,       # η_LTD (mV) - excitatory LTD
            
            # Inhibitory plasticity ratios (Table 2: η_r)
            'ETA_ILTP': 0.33,           # η_r for lateral inhibition LTP
            'ETA_ILTD': -0.33,          # η_r for lateral inhibition LTD
            
            # Normalization factors (Table 2: λ values)
            'NORM_FACTOR': 50,          # λ for excitatory (feedforward) connections
            'LATERAL_NORM_FACTOR': 6500,    # λ for lateral (distant) inhibition
            'TOPDOWN_NORM_FACTOR': 3500,    # λ for top-down inhibition
            'ETA_INH': 1500,            # λ for local inhibition
            
            # STDP configuration
            'STDP_LEARNING': "all",     # Enable all plasticity types
            'TRACKING': 'none',         # Tracking mode
            'SYNAPSE_DELAY': 0,         # Synaptic delay (ms)
            'DECAY_LEARNING': 0,        # Learning rate decay
            'TARGET_SPIKE_RATE': 0.75,  # Target spike rate (spikes/s)
            'MIN_THRESH': 4,            # Minimum threshold (mV)
            'ETA_SRA': 0,               # Spike rate adaptation learning rate
            'ETA_TA': 0,                # Threshold adaptation learning rate
        },
        "complex_cell_config": {
            # Membrane dynamics (Table 2)
            'VRESET': -10,              # V_reset (mV)
            'VTHRESH': 3,               # V_thresh (mV)
            'TAU_M': 50,                # τ_m (ms) - membrane time constant
            'TAU_RP': 5,                # τ_RP (ms) - refractory period time constant
            'TAU_LTP': 40,              # τ_LTP (ms) - LTP time constant
            'TAU_LTD': 40,              # τ_LTD (ms) - LTD time constant
            
            # Learning rates (Table 2)
            'ETA_RP': 5,                # η_RP (mV) - refractory period learning rate
            'ETA_LTP': 0.08,            # η_LTP (mV) - excitatory LTP
            'ETA_LTD': -0.08,           # η_LTD (mV) - excitatory LTD
            
            # Inhibitory plasticity ratios (Table 2: η_r)
            'ETA_ILTP': 0.25,           # η_r for local inhibition LTP
            'ETA_ILTD': -0.25,          # η_r for local inhibition LTD
            
            # Normalization factors (Table 2: λ values)
            'NORM_FACTOR': 1000,        # λ for excitatory connections
            'ETA_INH': 600,             # λ for local inhibition
            
            # STDP configuration
            'STDP_LEARNING': "all",     # Enable all plasticity types
            'TRACKING': 'none',         # Tracking mode
            'DECAY_LEARNING': 0,        # Learning rate decay
        }
    }
