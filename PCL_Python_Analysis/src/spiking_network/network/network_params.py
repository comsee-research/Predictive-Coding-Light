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
            'ETA_INH': [20],
            'ETA_LTD': [-0.0021],
            'ETA_LTP': [0.0077],
            'ETA_ILTP': [7.7],
            'ETA_ILTD': [-2.1],
            'ETA_RP': [1],
            'ETA_SRA': [0.6],
            'ETA_TA': [0],
            'MIN_THRESH': [4],
            'NORM_FACTOR': [4],
            'STDP_LEARNING': ["excitatory"],
            'SYNAPSE_DELAY': [0],
            'DECAY_LEARNING': [0],
            'TARGET_SPIKE_RATE': [0.75],
            'TAU_LTD': [14],
            'TAU_LTP': [7],
            'TAU_M': [18],
            'TAU_RP': [20],
            'TAU_SRA': [100],
            'TRACKING': ['partial'],
            'VRESET': [-20],
            'VTHRESH': [4]
        },
        "complex_cell_config": {
            'ETA_INH': [15],
            'ETA_LTD': [0.2],
            'ETA_LTP': [0.2],
            'ETA_RP': [1],
            'NORM_FACTOR': [10],
            'STDP_LEARNING': ["excitatory"],
            'DECAY_LEARNING': [0],
            'TAU_LTD': [20],
            'TAU_LTP': [20],
            'TAU_M': [20],
            'TAU_RP': [20],
            'TRACKING': ['partial'],
            'VRESET': [-20],
            'VTHRESH': [3]
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
