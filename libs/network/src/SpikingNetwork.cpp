//
// Created by Thomas on 14/04/2021.
//

#include "SpikingNetwork.hpp"
SpikingNetwork::SpikingNetwork() = default;
std::mt19937 generator_new2(time(nullptr));

/**
 * @brief Creates a spiking neural network.
 * It loads the configuration files from the specified networkPath.
 * Constructs the neuron connections and initializes them with random weights.
 * @param networkPath - Path to the network folder.
 */
SpikingNetwork::SpikingNetwork(const std::string &networkPath) : m_networkConf(NetworkConfig(networkPath + "configs/network_config.json")),
                                                                 m_simpleNeuronConf(networkPath + "configs/simple_cell_config.json", 0),
                                                                 m_complexNeuronConf(networkPath + "configs/complex_cell_config.json", 1),
                                                                 m_pixelMapping(std::vector<std::vector<uint64_t>>(
                                                                         m_networkConf.getVfWidth() * m_networkConf.getVfHeight(),
                                                                         std::vector<uint64_t>(0))),
                                                                 m_simpleWeightsOrientations(m_networkConf.getLayerConnectivity()[0].sizes[2], 
                                                                 std::vector<std::vector<int>>(3, std::vector<int>(0))),
                                                                 m_complexCellsOrientations(m_networkConf.getLayerConnectivity()[0].sizes[0]
                                                                 *m_networkConf.getLayerConnectivity()[0].sizes[1]
                                                                 *m_networkConf.getLayerConnectivity()[0].sizes[2], 
                                                                 std::vector<std::vector<int>>(3, std::vector<int>(0))),
                                                                 m_activation(true)
                                                                  {
    for (size_t i = 0; i < m_networkConf.getLayerConnectivity().size(); ++i) {
        addLayer(m_networkConf.getSharingType(), m_networkConf.getLayerConnectivity()[i]);
    }
}

/**
 * @brief Iterate the network on the event, updating every 1st layer neuron connected to the subsequent pixel.
 * Determines which neuron to update depending on a mapping between pixels and neurons.
 * If a neuron exceeds a threshold, it spikes and transmit another event towards deeper neurons.
 * A neuron spikes activates newLocalInhibitoryEvent connections to adjacent neurons.
 * @param event - The event coming from the pixel array.
 */
void SpikingNetwork::addEvent(const Event &event) {
    if (m_networkConf.getNeuron1Synapses() == 1) {
        auto height = m_networkConf.getVfHeight();
        for (size_t ind: m_pixelMapping[static_cast<uint32_t>(event.x()) * height +
                                       static_cast<uint32_t>(event.y())]) { 
            auto eventPos = Position(event.x() - static_cast<int16_t>(m_neurons[0][ind].get().getOffset().x()),
                                    event.y() - static_cast<int16_t>(m_neurons[0][ind].get().getOffset().y()));
            bool spiked = m_neurons[0][ind].get().newEvent(Event(event.timestamp(), 
            eventPos.x(), eventPos.y(), event.polarity(), event.camera()));
            double wi = m_neurons[0][ind].get().getWeightsMatrix().get(event.polarity(), 
            event.camera(), event.synapse(), eventPos.x(), eventPos.y());
            neuronsStatistics(event.timestamp(), 0, m_neurons[0][ind].get().getPos(), m_neurons[0][ind].get(), wi, spiked);
            if (spiked) {
                m_neurons[0][ind].get().weightUpdate();
                if(m_networkConf.getVfHeight()==1 || m_networkConf.getVfWidth()==1) {
                    normalizeWeights();
                    m_LatNorm = m_simpleNeuronConf.LATERAL_NORM_FACTOR;
                }
                lateralLocalInhibition(m_neurons[0][ind].get());
                if(m_activation) {
                   lateralDynamicInhibition(m_neurons[0][ind].get());
                }
                if (m_neurons.size() > 1) {
                  addNeuronEvent(m_neurons[0][ind].get());
                }
            }
        }
    } 
    else {
        m_lastEventTs = event.timestamp();
        for(size_t synapse = 0; synapse < m_networkConf.getNeuron1Synapses(); synapse++) { 
            Event ev(event.timestamp() + synapse * m_simpleNeuronConf.SYNAPSE_DELAY, event.x(), 
            event.y(), event.polarity(), event.camera(), synapse, event.over());
            m_eventsList.emplace(ev);
        }
        processSynapticEvent();
    }
}

/**
 * @brief Recursive function that updates neurons deeper than the 1st layer. Similarly to the addEvent function, if a neuron spikes,
 * it transmits a new event forward.
 * the neuron to which the event is transmitted are determined from vector of reference called outConnections.
 * A neuron spikes activates newLocalInhibitoryEvent connections to adjacent neurons.
 * @param neuron - The neuron that just spiked.
 */
inline void SpikingNetwork::addNeuronEvent(const Neuron &neuron) {
    for (auto &forwardNeuron: neuron.getOutConnections()) {
        bool spiked = forwardNeuron.get().newEvent(NeuronEvent(neuron.getSpikingTime(), 
        neuron.getPos().x()- static_cast<int16_t>(forwardNeuron.get().getOffset().x()),
        neuron.getPos().y()- static_cast<int16_t>(forwardNeuron.get().getOffset().y()), neuron.getPos().z(),
        neuron.getIndex(), neuron.getLayer()));
        double wi = 0;
        neuronsStatistics(neuron.getSpikingTime(), 0, forwardNeuron.get().getPos(), forwardNeuron.get(), wi, spiked);
        if (spiked) {
            forwardNeuron.get().weightUpdate();
            lateralLocalInhibition(forwardNeuron.get());
            if(m_activation) {
                topDownDynamicInhibition(forwardNeuron.get());
            }
            bool to_compute = false;
            if (forwardNeuron.get().getLayer() != m_complexNeurons.size()) {
                addNeuronEvent(forwardNeuron.get());
            }          
        }
    }
}

/**
 * Processes events in the case where there are synaptic delays.
 */
void SpikingNetwork::processSynapticEvent() {
    auto height = m_networkConf.getVfHeight();
    while(!m_eventsList.empty() && (m_eventsList.top().timestamp() <= m_lastEventTs || m_eventsList.top().over()) ) {
        Event event = m_eventsList.top();
        for (size_t ind: m_pixelMapping[static_cast<uint32_t>(event.x()) * height +
                                        static_cast<uint32_t>(event.y())]) {
            auto eventPos = Position(event.x() - static_cast<int16_t>(m_neurons[0][ind].get().getOffset().x()),
                                     event.y() - static_cast<int16_t>(m_neurons[0][ind].get().getOffset().y()));
            bool spiked = m_neurons[0][ind].get().newEvent(Event(event.timestamp(), 
            eventPos.x(), eventPos.y(), event.polarity(), event.camera(), event.synapse()));
            double wi = m_neurons[0][ind].get().getWeightsMatrix().get(event.polarity(),
            event.camera(), event.synapse(), eventPos.x(), eventPos.y());
            neuronsStatistics(event.timestamp(), 0, m_neurons[0][ind].get().getPos(), m_neurons[0][ind].get(), wi, spiked);
            if (spiked) {
                m_neurons[0][ind].get().weightUpdate();
                lateralLocalInhibition(m_neurons[0][ind].get());
                lateralDynamicInhibition(m_neurons[0][ind].get());
                if (m_neurons.size() > 1) {
                    addNeuronEvent(m_neurons[0][ind].get());
                }
            }
        }
        m_eventsList.pop();
    }
}

/**
 * @brief Propagation of an inhibitory event to neurons in the previous layer.
 * @param neuron - Neuron that triggered the event.
 */
void SpikingNetwork::topDownDynamicInhibition(Neuron &neuron) {
    for (auto &previousNeuron: neuron.getTopDownDynamicInhibitionConnections()) {
        auto event = NeuronEvent(neuron.getSpikingTime(), neuron.getPos().x(), 
        neuron.getPos().y(), neuron.getPos().z(), neuron.getIndex(), neuron.getLayer());
        previousNeuron.get().newTopDownInhibitoryEvent(event);
        Position pos(neuron.getPos().x(), neuron.getPos().y(),neuron.getIndex());
        neuronsStatistics(event.timestamp(), 3, pos, previousNeuron.get(), 
        previousNeuron.get().getTopDownInhibitionWeights(event.id()));
    }
}

/**
 * @brief Propagation of an inhibitory event to lateral neurons in the same layer.
 * @param neuron - Neuron that triggered the event.
 */
void SpikingNetwork::lateralDynamicInhibition(Neuron &neuron) {
    for (auto &lateralNeuron: neuron.getLateralDynamicInhibitionConnections()) {
        auto event = NeuronEvent(neuron.getSpikingTime(), neuron.getPos().x(), neuron.getPos().y(), neuron.getPos().z(), 
        neuron.getIndex(), neuron.getLayer());
        lateralNeuron.get().newLateralInhibitoryEvent(event);
        neuronsStatistics(event.timestamp(), 2, neuron.getPos(), lateralNeuron.get(), 
        lateralNeuron.get().getlateralInhibitionWeights(event.id()));
    }
}

/**
 * @brief Propagation of an inhibitory event to neurons with similar visual fields and in the same layer.
 * This event is not stored because there is no plasticity.
 * @param neuron - Neuron that triggered the event.
 */
void SpikingNetwork::lateralLocalInhibition(Neuron &neuron) {
    for (auto &lateralNeuron: neuron.getLateralLocalInhibitionConnections()) {
        auto event = NeuronEvent(neuron.getSpikingTime(), neuron.getPos().x(), 
        neuron.getPos().y(), neuron.getPos().z(), neuron.getPos().z(), neuron.getLayer());
        lateralNeuron.get().newLocalInhibitoryEvent(event);
        neuronsStatistics(neuron.getSpikingTime(), 1, neuron.getPos(), lateralNeuron.get(), lateralNeuron.get().getConf().ETA_INH);
    }
}

/**
 * @brief Generate all the weight matrices that will be used by neurons in the network.
 * The number of weight matrices depends on the type of weight sharing used (only for simple cells, 1st layer at the moment).
 * "patch": weight matrices are shared among neurons of similar depth (z axis) and of the same patch.
 * @param connections
 * @param nbNeurons
 */
void SpikingNetwork::generateWeightSharing(const LayerConnectivity &connections, const size_t nbNeurons) {
    size_t x = static_cast<long>(connections.neuronSizes[0][0]);
    size_t y = static_cast<long>(connections.neuronSizes[0][1]);
    size_t z = static_cast<long>(connections.neuronSizes[0][2]);
    if (connections.neuronType == "SimpleCell") {
        std::vector<size_t> dimensions;
        
        if( (m_networkConf.getVfHeight() == 1 && connections.sizes[1] == 1) || (m_networkConf.getVfWidth() == 1 && 
            connections.sizes[0] == 1) ) {
            size_t POL = 1;
            dimensions = std::vector<size_t>({POL, static_cast<size_t>(m_networkConf.getNbCameras()),
                                               static_cast<size_t>(m_networkConf.getNeuron1Synapses()), x, y});
        }
        else {
            dimensions = std::vector<size_t>({NBPOLARITY, static_cast<size_t>(m_networkConf.getNbCameras()),
                                               static_cast<size_t>(m_networkConf.getNeuron1Synapses()), x, y});
        }
        auto dimensions_inhib = std::vector<size_t>({static_cast<long>(connections.sizes[2])}); 


        if (m_networkConf.getSharingType() == "patch") {
            size_t patch_size = connections.patches[0].size() * connections.patches[1].size();
            for (size_t patch = 0; patch < patch_size; ++patch) {
                for (size_t i = 0; i < connections.sizes[2]; ++i) {
                    m_sharedWeightsSimple.emplace_back(dimensions, true, m_simpleNeuronConf.NORM_FACTOR);
                }
            }
            for (size_t i = 0; i < connections.sizes[2]; ++i) { 
                m_sharedInhibWeightsSimple.emplace_back(dimensions_inhib, true, m_simpleNeuronConf.ETA_INH);
            }
        } else {
            for (size_t i = 0; i < nbNeurons; ++i) {
                m_sharedWeightsSimple.emplace_back(dimensions, false, m_simpleNeuronConf.NORM_FACTOR);
                m_sharedInhibWeightsSimple.emplace_back(dimensions_inhib, true, m_simpleNeuronConf.ETA_INH);
            }
            std::cout << "No weight sharing: basic predictive example" << std::endl;
        }
        
    }
    else {
        auto dimensions = connections.neuronSizes[0];
        auto dimensions_inhib = std::vector<size_t>({static_cast<long>(connections.sizes[2])}); 
        std::vector<WeightMatrix> temp;
        m_sharedWeightsComplex.emplace_back(temp);
        m_sharedInhibWeightsComplex.emplace_back(temp);

        if (m_networkConf.getSharingType() == "patch") {
            size_t patch_size = connections.patches[0].size() * connections.patches[1].size();
            for (size_t patch = 0; patch < patch_size; ++patch) {
                for (size_t i = 0; i < connections.sizes[2]; ++i) {
                    m_sharedWeightsComplex[connections.interConnections[0]].emplace_back(dimensions, true, 
                    m_complexNeuronConf.NORM_FACTOR);
                }
            }
            for (size_t i = 0; i < connections.sizes[2]; ++i) { 
                m_sharedInhibWeightsComplex[connections.interConnections[0]].emplace_back(dimensions_inhib, true, 
                m_complexNeuronConf.ETA_INH);
            }

        } else {
            for (size_t i = 0; i < nbNeurons; ++i) {
                m_sharedWeightsComplex[connections.interConnections[0]].emplace_back(dimensions, true, m_complexNeuronConf.NORM_FACTOR);
                m_sharedInhibWeightsComplex[connections.interConnections[0]].emplace_back(dimensions_inhib, true, 
                m_complexNeuronConf.ETA_INH);
            }
            std::cout << "No weight sharing: basic predictive example" << std::endl;
        }
    }
}

/**
 *
 * @param sgaringType
 * @param connections
 */
void SpikingNetwork::addLayer(const std::string &sharingType, const LayerConnectivity &connections) {
    auto nbNeurons = connections.patches[0].size() * connections.sizes[0] * connections.patches[1].size() * 
                     connections.sizes[1] * connections.patches[2].size() * connections.sizes[2];
    generateWeightSharing(connections, nbNeurons);
    size_t neuronIndex = 0;
    size_t weightIndex;
    size_t countWeightSharing = 0;
    m_layout.emplace_back();
    auto layer = m_neurons.size();
    std::cout << "layer = " << layer << std::endl;
    auto depth = std::vector<size_t>({1, 1, static_cast<size_t>(connections.sizes[2])});
    if (connections.neuronType == "ComplexCell") {
        std::vector<ComplexNeuron> temp;
        m_complexNeurons.emplace_back(temp);
    }

    for (size_t x = 0; x < connections.patches[0].size(); ++x) {
        for (size_t y = 0; y < connections.patches[1].size(); ++y) {
            for (size_t z = 0; z < connections.patches[2].size(); ++z) {
                for (size_t i = 0; i < connections.sizes[0]; ++i) {
                    for (size_t j = 0; j < connections.sizes[1]; ++j) {
                        for (size_t k = 0; k < connections.sizes[2]; ++k) {
                            if (sharingType == "patch") {
                                weightIndex = countWeightSharing * connections.sizes[2] + k;
                            }
                            else {
                                weightIndex = neuronIndex;
                            } 

                            // Position of the neuron in the neuronal layer space (x, y, z)
                            auto pos = Position(x * connections.sizes[0] + i, y * connections.sizes[1] + j, 
                                       z * connections.sizes[2] + k);

                            // Receptive field starting position (x, y) compared to the previous layer
                            auto offset = Position(connections.patches[0][x] + i * 
                                                  (connections.neuronSizes[0][0] - connections.neuronOverlap[0]),
                                                   connections.patches[1][y] + j *
                                                  (connections.neuronSizes[0][1] - connections.neuronOverlap[1]));

                            if (connections.neuronType == "SimpleCell") {
                                m_simpleNeurons.emplace_back(SimpleNeuron(neuronIndex, layer, m_simpleNeuronConf, pos, offset,
                                                            m_sharedWeightsSimple[weightIndex], 
                                                            m_sharedInhibWeightsSimple[weightIndex], 
                                                            m_networkConf.getNeuron1Synapses(), m_LatNorm));
                                m_simpleNeurons.at(m_simpleNeurons.size()-1).setInhibitionRange(m_networkConf.getNeuronInhibitionRange());
                            } else if (connections.neuronType == "ComplexCell") {
                                m_complexNeurons[connections.interConnections[0]].emplace_back(ComplexNeuron(neuronIndex, layer, 
                                m_complexNeuronConf, pos, offset, 
                                connections.neuronSizes[0], depth,
                                m_sharedWeightsComplex[connections.interConnections[0]][weightIndex], 
                                m_sharedInhibWeightsComplex[connections.interConnections[0]][weightIndex]));
                            } else {
                                std::cout << "No matching cell type" << std::endl;
                            }
                            m_layout[layer][{pos.x(), pos.y(), pos.z()}] = neuronIndex;
                            ++neuronIndex;
                        }
                    }
                }
                if (sharingType == "patch") {
                    ++countWeightSharing;
                }
            }
        }
    }

    if (connections.neuronType == "SimpleCell") {
        m_neurons.emplace_back(m_simpleNeurons.begin(), m_simpleNeurons.end());
    } else if (connections.neuronType == "ComplexCell") {
        m_neurons.emplace_back(m_complexNeurons[connections.interConnections[0]].begin(), 
        m_complexNeurons[connections.interConnections[0]].end());
    } else {
        std::cout << "No matching cell type" << std::endl;
    }
    m_structure.push_back(m_neurons[layer].size());
    connectLayer(connections);
    normalizeWeights();
    m_LatNorm = m_simpleNeuronConf.LATERAL_NORM_FACTOR;
}

/**
 *
 * @param connections
 */
void SpikingNetwork::connectLayer(const LayerConnectivity &connections) {
    auto currLayer = m_neurons.size() - 1;
    for (auto &neuron: m_neurons[currLayer]) {
        topDownConnection(neuron.get(), connections.interConnections, currLayer, connections.neuronSizes, connections.inhibitions);
        if (std::find(connections.inhibitions.begin(), connections.inhibitions.end(), "local") != connections.inhibitions.end()) {
            lateralLocalInhibitionConnection(neuron.get(), currLayer, connections.sizes);
        }
        if (std::find(connections.inhibitions.begin(), connections.inhibitions.end(), "lateral") != connections.inhibitions.end()) {
            lateralDynamicInhibitionConnection(neuron.get(), currLayer, connections.patches, connections.sizes);
        }
    }
}

/**
 *
 * @param neuron
 * @param currLayer
 * @param inhibition
 * @param interConnections
 * @param neuronSizes
 */
void SpikingNetwork::topDownConnection(Neuron &neuron, const std::vector<int> &interConnections, 
                                       const size_t currLayer, const std::vector<std::vector<size_t>>
                                       &neuronSizes, const std::vector<std::string> &inhibition) {
    if (interConnections.size() > 1) { // If there is a skip connection, all to all connection is forced.
        for (const auto layerToConnect: interConnections) {
            for (const auto &prevNeuron: m_neurons[layerToConnect]) {
                neuron.addInConnection(prevNeuron);
                prevNeuron.get().addOutConnection(neuron);
            }
        }
    } else {
        const auto layerToConnect = interConnections[0];
        const auto &neuronSize = neuronSizes[0];
        for (size_t i = neuron.getOffset().x(); i < neuron.getOffset().x() + neuronSize[0]; ++i) {
            for (size_t j = neuron.getOffset().y(); j < neuron.getOffset().y() + neuronSize[1]; ++j) {
                for (size_t k = neuron.getOffset().z(); k < neuron.getOffset().z() + neuronSize[2]; ++k) {
                    if (currLayer == 0) {
                        m_pixelMapping[i * m_networkConf.getVfHeight() + j].push_back(neuron.getIndex());
                    } else {
                        neuron.addInConnection(m_neurons[layerToConnect][m_layout[layerToConnect][{i, j, k}]]);
                        m_neurons[layerToConnect][m_layout[layerToConnect][{i, j, k}]].get().addOutConnection(neuron);
                        if ((std::find(inhibition.begin(), inhibition.end(), "topdown") != inhibition.end())) {
                            neuron.addTopDownDynamicInhibitionConnection(m_neurons[layerToConnect][m_layout[layerToConnect][{i, j, k}]]);
                            m_neurons[layerToConnect][m_layout[layerToConnect][{i, j, k}]].get().initTopDownDynamicInhibitionWeights(neuron.getIndex());
                            m_neurons[layerToConnect][m_layout[layerToConnect][{i, j, k}]].get().initTopDownDynamicExcitationWeights(neuron.getIndex());
                            m_neurons[layerToConnect][m_layout[layerToConnect][{i, j, k}]].get().addTopDownDynamicInhibitionConnection(neuron);
                        }
                    }
                }
            }
        }
    }
}

/**
 *
 * @param neuron
 * @param currLayer
 * @param layerPatches
 * @param layerSizes
 */
void SpikingNetwork::lateralDynamicInhibitionConnection(Neuron &neuron, const size_t currLayer,
                                                        const std::vector<std::vector<size_t>> &layerPatches,
                                                        const std::vector<size_t> &layerSizes) { // TODO: optim -> very slow
    for (int x = static_cast<int>(neuron.getPos().x() - m_networkConf.getNeuronInhibitionRange().at(0)); 
         x < static_cast<int>(neuron.getPos().x() + m_networkConf.getNeuronInhibitionRange().at(0)+1); ++x) {
        for (int y = static_cast<int>(neuron.getPos().y() - m_networkConf.getNeuronInhibitionRange().at(1)); 
            y < static_cast<int>(neuron.getPos().y() + m_networkConf.getNeuronInhibitionRange().at(1) + 1); ++y) {
            for (size_t z = 0; z < layerSizes[2]; ++z) {
                if ((x != neuron.getPos().x() || y != neuron.getPos().y()) && x >= 0 && y >= 0 &&
                    x < layerPatches[0].size() * layerSizes[0] &&
                    y < layerPatches[1].size() * layerSizes[1]) {   
                    if(m_networkConf.getVfHeight() == 1 || m_networkConf.getVfWidth() == 1) {
                        if( (neuron.getPos().x() == 0 && neuron.getPos().y() == 0) || (x == 0 && y == 0) ) {
                            neuron.addLateralDynamicInhibitionConnections(m_neurons[currLayer][m_layout[currLayer][{x, y, z}]]);
                            neuron.initLateralDynamicInhibitionWeights(m_neurons[currLayer][m_layout[currLayer][{x, y, z}]].get().getIndex());
                            neuron.initLateralDynamicExcitationWeights(m_neurons[currLayer][m_layout[currLayer][{x, y, z}]].get().getIndex());
                            if(neuron.getPos().x() == 0 && neuron.getPos().y() == 0) {
                                neuron.assignTonoLat(true);
                            }
                            else {
                                m_LatNorm += neuron.getlateralInhibitionWeights(m_neurons[currLayer][m_layout[currLayer][{x, y, z}]].get().getIndex());
                            }
                        }
                    }
                    else {
                        neuron.addLateralDynamicInhibitionConnections(m_neurons[currLayer][m_layout[currLayer][{x, y, z}]]);
                        neuron.initLateralDynamicInhibitionWeights(m_neurons[currLayer][m_layout[currLayer][{x, y, z}]].get().getIndex());
                        neuron.initLateralDynamicExcitationWeights(m_neurons[currLayer][m_layout[currLayer][{x, y, z}]].get().getIndex());
                    }                                             
                }
            }
        }
    }
}

/**
 *
 * @param neuron
 * @param currLayer
 * @param layerSizes
 */
void SpikingNetwork::lateralLocalInhibitionConnection(Neuron &neuron, const size_t currLayer,
                                                       const std::vector<size_t> &layerSizes) {
    for (size_t z = 0; z < layerSizes[2]; ++z) {
        if (z != neuron.getPos().z()) {
            neuron.addLateralLocalInhibitionConnections(m_neurons[currLayer][m_layout[currLayer][{neuron.getPos().x(), 
                                                         neuron.getPos().y(), z}]]);
        }
    }
}

void SpikingNetwork::normalizeWeights() {
    for (auto &neurons: m_neurons) {
        for (auto &neuron: neurons) {
            neuron.get().normalizeL1Weights();
        }
    }
}

/**
 * Updates some neurons internal state.
 * Computes each neurons average spiking rate.
 * Adapts the threshold of the 1st layer's neurons.
 * @param timeInterval
 */
void SpikingNetwork::updateNeuronsStates(long timeInterval) {
    size_t layer = 1;
    double averageActivity = 0;
    for (auto &neurons: m_neurons) {
        for (auto &neuron: neurons) {
            neuron.get().updateState(timeInterval);
            neuron.get().thresholdAdaptation();
            if (layer == 1) {
                averageActivity += neuron.get().getSpikingRate();
            }
        }
    }
    averageActivity /= static_cast<double>(m_neurons[0].size());
    auto alpha = 0.6;
    m_averageActivity = (alpha * averageActivity) + (1.0 - alpha) * m_averageActivity;
}

/**
 *
 */
void SpikingNetwork::saveNetwork() {
    saveNeuronsStates();
    saveNetworkLayout();
}

/**
 *
 */
void SpikingNetwork::saveNetworkLayout() {
    size_t count;
    for (size_t layer = 0; layer < m_neurons.size(); ++layer) {
        std::vector<size_t> data(static_cast<size_t>(getNetworkStructure()[layer]));
        count = 0;
        auto size_x = m_networkConf.getLayerConnectivity()[layer].patches[0].size() * 
                      m_networkConf.getLayerConnectivity()[layer].sizes[0];
        auto size_y = m_networkConf.getLayerConnectivity()[layer].patches[1].size() * 
                      m_networkConf.getLayerConnectivity()[layer].sizes[1];
        auto size_z = m_networkConf.getLayerConnectivity()[layer].patches[2].size() * 
                      m_networkConf.getLayerConnectivity()[layer].sizes[2];
        for (size_t i = 0; i < size_x; ++i) {
            for (size_t j = 0; j < size_y; ++j) {
                for (size_t k = 0; k < size_z; ++k) {
                    data[count] = m_layout[layer][{i, j, k}];
                    ++count;
                }
            }
        }
        cnpy::npy_save(m_networkConf.getNetworkPath() + "weights/layout_" + std::to_string(layer) + ".npy", &data[0],
                       {size_x, size_y, size_z}, "w");
    }
}

/**
 *
 * @param saveCount
 */
void SpikingNetwork::intermediateSave(size_t saveCount) {
    std::string fileName;
    fs::create_directory(m_networkConf.getNetworkPath() + "weights/intermediate_" + std::to_string(saveCount) + "/");
    fs::create_directory(m_networkConf.getNetworkPath() + "weights/intermediate_" + std::to_string(saveCount) + "/0/");
    fs::create_directory(m_networkConf.getNetworkPath() + "weights/intermediate_" + std::to_string(saveCount) + "/1/");

    size_t layer = 0;
    bool tdi, li;

    for (auto &neurons: m_neurons) {
        if (std::find(m_networkConf.getLayerConnectivity()[layer].inhibitions.begin(),
                      m_networkConf.getLayerConnectivity()[layer].inhibitions.end(), "lateral") != m_networkConf.getLayerConnectivity()[layer].inhibitions.end()) {
            li = true;
        } else {
            li = false;
        }
        if (layer+1 != m_networkConf.getLayerConnectivity().size() && std::find(m_networkConf.getLayerConnectivity()[layer+1].inhibitions.begin(),
                      m_networkConf.getLayerConnectivity()[layer+1].inhibitions.end(), "topdown") != m_networkConf.getLayerConnectivity()[layer+1].inhibitions.end()) {
            tdi = true;
        } else {
            tdi = false;
        }

        if (layer == 0) {
            for (size_t i = 0; i < m_networkConf.getLayerConnectivity()[layer].sizes[2]; ++i) {
                fileName = m_networkConf.getNetworkPath() + "weights/intermediate_" + std::to_string(saveCount) + "/" + std::to_string(layer) + "/";
                neurons[i].get().saveWeights(fileName);
                neurons[i].get().savePlasticLocalInhibitionWeights(fileName);
            }
        } 
        for (auto &neuron: neurons) {
            fileName = m_networkConf.getNetworkPath() + "weights/intermediate_" + std::to_string(saveCount) + "/" + std::to_string(layer) + "/";
            neuron.get().saveState(fileName);
            if (li) {
                neuron.get().saveLateralInhibitionWeights(fileName);
            }
            if (tdi) {
                neuron.get().saveTopDownInhibitionWeights(fileName);
            }
            if (layer != 0 || m_networkConf.getSharingType() != "patch") {
                neuron.get().saveWeights(fileName);
                if(layer == 0) {
                    neuron.get().savePlasticLocalInhibitionWeights(fileName);
                }
            }
        }
        ++layer;
    }
}

/**
 *
 */
void SpikingNetwork::saveNeuronsStates() {
    size_t layer = 0;
    std::string fileName;
    bool tdi, li;

    for (auto &neurons: m_neurons) {
        if (std::find(m_networkConf.getLayerConnectivity()[layer].inhibitions.begin(),
                      m_networkConf.getLayerConnectivity()[layer].inhibitions.end(), "lateral") != 
                      m_networkConf.getLayerConnectivity()[layer].inhibitions.end()) {
            li = true;
        } else {
            li = false;
        }
        if (std::find(m_networkConf.getLayerConnectivity()[layer+1].inhibitions.begin(),
                      m_networkConf.getLayerConnectivity()[layer+1].inhibitions.end(), "topdown") != 
                      m_networkConf.getLayerConnectivity()[layer+1].inhibitions.end()) {
            tdi = true;
        } else {
            tdi = false;
        }

        if (layer == 0 && m_networkConf.getSharingType() == "patch") {
            size_t step = m_networkConf.getLayerConnectivity()[layer].sizes[0] * 
                          m_networkConf.getLayerConnectivity()[layer].sizes[1] * 
                          m_networkConf.getLayerConnectivity()[layer].sizes[2];
            size_t patch_size = m_networkConf.getLayerConnectivity()[layer].patches[0].size() * 
                                m_networkConf.getLayerConnectivity()[layer].patches[1].size();
            for (size_t patch = 0; patch < patch_size; ++patch) {
                for (size_t i = 0; i < m_networkConf.getLayerConnectivity()[layer].sizes[2]; ++i) {
                    fileName = m_networkConf.getNetworkPath() + "weights/" + std::to_string(layer) + "/";
                    neurons[patch * step + i].get().saveWeights(fileName);
                    neurons[patch * step + i].get().savePlasticLocalInhibitionWeights(fileName);
                }
            }
        } 
        for (auto &neuron: neurons) {
            fileName = m_networkConf.getNetworkPath() + "weights/" + std::to_string(layer) + "/";
            neuron.get().saveState(fileName);
            if (li) {
                neuron.get().saveLateralInhibitionWeights(fileName);
            }
            if (tdi) {
                neuron.get().saveTopDownInhibitionWeights(fileName);
            }
            if (layer != 0 || m_networkConf.getSharingType() != "patch") {
                neuron.get().saveWeights(fileName);
                if(layer == 0) {
                    neuron.get().savePlasticLocalInhibitionWeights(fileName);
                }
            }
        }
        ++layer;
    }
}

/**
 *
 */
void SpikingNetwork::loadWeights() {
    std::string statePath, filePath, tdiFilePath, liFilePath;
    size_t layer = 0;
    bool tdi, li;
    cnpy::npz_t liArrayNPZ, tdiArrayNPZ;

    for (auto &neurons: m_neurons) {
        std::string path(m_networkConf.getNetworkPath() + "weights/" + std::to_string(layer) + "/0.npy");
        liFilePath = m_networkConf.getNetworkPath() + "weights/" + std::to_string(layer) + "/0li.npy";
        tdiFilePath = m_networkConf.getNetworkPath() + "weights/" + std::to_string(layer) + "/0tdi.npy";
        if (std::find(m_networkConf.getLayerConnectivity()[layer].inhibitions.begin(), 
                      m_networkConf.getLayerConnectivity()[layer].inhibitions.end(), "lateral") !=
                      m_networkConf.getLayerConnectivity()[layer].inhibitions.end() && Util::fileExist(liFilePath)) {
            li = true;
        } else {
            li = false;
        }
        if (std::find(m_networkConf.getLayerConnectivity()[layer+1].inhibitions.begin(), 
                      m_networkConf.getLayerConnectivity()[layer+1].inhibitions.end(), "topdown") !=
                      m_networkConf.getLayerConnectivity()[layer+1].inhibitions.end() && Util::fileExist(tdiFilePath)) {
            tdi = true;
        } else {
            tdi = false;
        }
        if (Util::fileExist(path)) {
            if (layer == 0 && m_networkConf.getSharingType() == "patch") {
                size_t step = m_networkConf.getLayerConnectivity()[layer].sizes[0] * 
                              m_networkConf.getLayerConnectivity()[layer].sizes[1] * 
                              m_networkConf.getLayerConnectivity()[layer].sizes[2];
                size_t patch_size = m_networkConf.getLayerConnectivity()[layer].patches[0].size() * 
                                    m_networkConf.getLayerConnectivity()[layer].patches[1].size();
                for (size_t patch = 0; patch < patch_size; ++patch) {
                    for (size_t i = 0; i < m_networkConf.getLayerConnectivity()[layer].sizes[2]; ++i) {
                        filePath = m_networkConf.getNetworkPath() + "weights/" + std::to_string(layer) + "/";
                        neurons[patch * step + i].get().loadWeights(filePath);
                        neurons[patch * step + i].get().loadPlasticLocalInhibitionWeights(filePath);
                    }
                }
            } 
            for (auto &neuron: neurons) {
                filePath = m_networkConf.getNetworkPath() + "weights/" + std::to_string(layer) + "/";
                neuron.get().loadState(filePath);
                if (li) {
                    neuron.get().loadLateralInhibitionWeights(filePath);
                }
                if (tdi) {
                    neuron.get().loadTopDownInhibitionWeights(filePath);
                }
                if (layer != 0 || m_networkConf.getSharingType() != "patch") {
                    neuron.get().loadWeights(filePath);
                    if(layer == 0) { 
                        neuron.get().loadPlasticLocalInhibitionWeights(filePath);
                    }
                }
            }
            std::cout << "Layer " << layer << ": weights loaded from file" << std::endl;
        } else {
            std::cout << "Layer " << layer << ": new weights generated" << std::endl;
        }
        ++layer;
    }
}

/**
 *
 * @param index
 * @param layer
 * @return
 */
std::reference_wrapper<Neuron> &SpikingNetwork::getNeuron(const size_t index, const size_t layer) {
    if (layer < m_neurons.size()) {
        if (index < m_neurons[layer].size()) {
            return m_neurons[layer][index];
        }
    }
    throw std::runtime_error("Wrong layer or index for neuron selection");
}

void SpikingNetwork::resetAllNeurons() {
    if(m_eventsParameters.size()!=0) {
        computeBinTS();
    }

    int n_spikes = 0;
    for (auto &simpleNeuron: m_simpleNeurons) {
        n_spikes += simpleNeuron.getOutputSize();
        simpleNeuron.resetNeuron();
        simpleNeuron.resetSimpleNeuron();
    }
    for (auto &complexNeuronLayer: m_complexNeurons) {
        for (auto &complexNeuron: complexNeuronLayer) {
            complexNeuron.resetNeuron();
            complexNeuron.resetComplexNeuron();
        }
    }
}

/**
 * Updates the statistics of a tracked neuron everytime it receives an event.
 * @param time 
 * @param type_ 
 * @param pos 
 * @param neuron 
 * @param wi 
 * @param spike 
 */
void SpikingNetwork::neuronsStatistics(uint64_t time, int type_, Position pos, Neuron &neuron, double wi, bool spike, 
                                       bool recordAllPotentials) {
    if(true) 
    // if ( (neuron.getConf().POTENTIAL_TRACK[0] == neuron.getPos().x() && neuron.getConf().POTENTIAL_TRACK[1] == neuron.getPos().y()) ) //|| // ||
        {
         neuron.assignToAmountOfEvents(type_);
        
        if(spike) {
        //     if((neuron.getConf().POTENTIAL_TRACK[0]==2) && (neuron.getConf().POTENTIAL_TRACK[1]==3)) {
        // }
            neuron.assignToPotentialTrain(std::make_pair(neuron.getSpikingPotential(), time));
            neuron.assignToPotentialThreshold();
        }
        if(recordAllPotentials) {
            neuron.assignToPotentialThreshold();
            neuron.assignToPotentialTrain(std::make_pair(neuron.getPotential(time), time));
        }
        if (type_ != 0) { //add a type
            type_ -= 1;
            // to change at some point so that the depth is automatically given
            // if (type_ == 1) {
            // //    neuron.assignToSumLateralWeights(type_, pos, wi, 121); // m_networkConf.getLayerConnectivity()[0].sizes()[2]);
            // }
            // if(type_ == 2) {
            // //    neuron.assignToSumTopDownWeights(pos.z(),wi,  32); //m_networkConf.getLayerConnectivity()[1].sizes()[2]);
            // }
            // if(type_ == 3) {
            // //    neuron.assignToSumLateralExcitatoryWeights(type_, pos, wi, 64);
            // }
            // if(type_ == 4) {
            // //    neuron.assignToSumTopDownExcitatoryWeights(pos.z(),wi, 16);
            // }
            if(recordAllPotentials) {
                std::tuple<double, double, uint64_t> var = {neuron.getBeforeInhibitionPotential(), neuron.getPotential(time), time};
                neuron.assignToTimingOfInhibition(type_, var);
            }
        }
    }
}

/**
 * Saves the statistics of all tracked neurons while making sure to erase potential old statistics that exists in the folder.
 * @param simulation 
 * @param sequence 
 */
void SpikingNetwork::saveStatistics(int simulation, int sequence, const std::string& folderName, bool sep_speed, int max_speed) {
    std::string fileName;
    size_t layer = 0;
    fs::file_status s = fs::file_status{};
    static bool entered = false;
    static std::string rememberFolderName;
    std::string newName = folderName;
    static int it = 0;
    if(sep_speed) {
        newName.erase(newName.begin()+8);
        newName.erase(newName.begin()+8);
        if(rememberFolderName!= newName && it==max_speed) {
            it = 0;
        }
    }
    
    if(!sep_speed) {
        if((!rememberFolderName.empty() && rememberFolderName != newName)) {
            entered = false;
        }
    }

    else {
        if((!rememberFolderName.empty() && rememberFolderName != newName) || it<max_speed) {
            it+=1;
            entered = false;
        }
    }

    for (auto &neurons: m_neurons) {
        if(!entered) {
            fs::remove_all(m_networkConf.getNetworkPath() + "statistics/" + std::to_string(layer) + "/" + folderName);
            fs::create_directory(
                        m_networkConf.getNetworkPath() + "statistics/" + std::to_string(layer) + "/" + folderName);
            
            fs::remove_all(m_networkConf.getNetworkPath() + "statistics/" + std::to_string(layer+1) + "/" + folderName);
            fs::create_directory(
                        m_networkConf.getNetworkPath() + "statistics/" + std::to_string(layer+1) + "/" + folderName);
            entered = true;
            rememberFolderName = folderName;
            if(sep_speed) {
                rememberFolderName.erase(rememberFolderName.begin()+8);
                rememberFolderName.erase(rememberFolderName.begin()+8);
            }
        }
        std::string path(m_networkConf.getNetworkPath() + "statistics/" + std::to_string(layer));
        if (fs::status_known(s) ? fs::exists(s) : fs::exists(path)) {
            fs::create_directory(
                    m_networkConf.getNetworkPath() + "statistics/" + std::to_string(layer) + "/"  + 
                    folderName + std::to_string(simulation));
            fs::create_directory(
                    m_networkConf.getNetworkPath() + "statistics/" + std::to_string(layer) + "/" + 
                    folderName + std::to_string(simulation) + "/" + std::to_string(sequence));


            for (auto &neuron: neurons) {
                if(true) 
                // if ( (neuron.get().getConf().POTENTIAL_TRACK[0] == neuron.get().getPos().x() 
                // && neuron.get().getConf().POTENTIAL_TRACK[1] == neuron.get().getPos().y()) )  
                  {
                    fileName = m_networkConf.getNetworkPath() + "statistics/" + std::to_string(layer) + 
                    "/" + folderName + std::to_string(simulation) + "/" + std::to_string(sequence) + "/";
                    saveStatesStatistics(fileName, neuron.get());
                }
            }
            ++layer;
        }
    }
}

/**
 * Writes the json file of the statistics of a neuron.
 * @param state 
 * @param neuron 
 */
void SpikingNetwork::writeJsonNeuronsStatistics(nlohmann::json &state, Neuron &neuron) {
    state["amount_of_events"] = neuron.getAmountOfEvents();
    state["potentials_thresholds"] = neuron.getPotentialThreshold();
    state["potential_train"] = neuron.getPotentialTrain();
    state["sum_inhib_weights"] = neuron.getSumLateralWeights();
    state["sum_topdown_weights"] = neuron.getSumTopDownWeights();
    state["timing_of_inhibition"] = neuron.getTimingOfInhibition();
    state["excitatory_ev"] = neuron.getExcitatoryEvents();
}

void SpikingNetwork::saveStatesStatistics(std::string &fileName, Neuron &neuron) {
    nlohmann::json state;
    writeJsonNeuronsStatistics(state, neuron);
    std::ofstream ofs(fileName + std::to_string(neuron.getIndex()) + ".json");
    if (ofs.is_open()) {
        ofs << std::setw(4) << state << std::endl;
    } else {
        std::cout << "cannot save neuron state statistics file" << std::endl;
    }
    ofs.close();
}

/**
 * Change the coordinates of the neuron to track regardless of what is in the config file.
 * @param n_x 
 * @param n_y 
 */
void SpikingNetwork::changeTrack(int n_x, int n_y) {
    std::vector<int> val{n_x, n_y};
    int layer=0;
    for (auto &neurons: m_neurons) {
            for (auto &neuron: neurons) {
                neuron.get().setPotentialTrack(val);
            }
            ++layer;
            if(layer!=0)
            {
                break;
            }
    }
}

/**
 * Re-initializes the lateral inhibition weights to random values sampled from an uniform distribution.
 */
void SpikingNetwork::randomLateralInhibition(int norm_factor) {
//    for (auto &neurons: m_neurons) {
        for (auto &neuron: m_neurons[0]) {
                neuron.get().randomInhibition(norm_factor);
            }
    }
//}
/**
 * Shuffles a specific type or all inhibitory weights.
 * @param cases 
 */
void SpikingNetwork::shuffleInhibition(int cases) {
    switch (cases) {
        case 1:
            for (auto &neurons: m_neurons[0]) {
                neurons.get().shuffleLateralInhibition();
            }
            break;
        case 2:
            for (auto &neurons: m_neurons[0]) {
                neurons.get().shuffleTopDownInhibition();
            }
            break;
        case 3:
            for (auto &neurons: m_neurons[0]) {
                neurons.get().shuffleLateralInhibition();
                neurons.get().shuffleTopDownInhibition();
            }
            break;
        default:
            std::cout << "No argument passed in \"shuffleInhibition()\". Function aborted." << std::endl;
            break;
    } 
}

/**
 * Assigns an orientation to a simple cell.
 * @param index_z 
 * @param orientation 
 */
void SpikingNetwork::assignOrientations(int index_z, int orientation, int thickness) {
    m_simpleWeightsOrientations.at(index_z).at(thickness).push_back(orientation);
}

/**
 * Assigns an orientation to a complex cell.
 * @param id 
 * @param orientation 
 */
void SpikingNetwork::assignComplexOrientations(int id, int orientation, int thickness) {
    m_complexCellsOrientations.at(id).at(thickness).push_back(orientation);
}

/**
 * Saves the orientation of simple and complex cells in a json file.
 */
void SpikingNetwork::saveOrientations() {
    std::string fileName;
    size_t layer = 0;
    fs::file_status s = fs::file_status{};
    std::string path(m_networkConf.getNetworkPath() + "statistics/");
    if (fs::status_known(s) ? fs::exists(s) : fs::exists(path)) {
        fs::create_directory(m_networkConf.getNetworkPath() + "statistics/" + "orientations");
        fileName = m_networkConf.getNetworkPath() + "statistics/" + "orientations" + "/";
        nlohmann::json state;
        state["orientations"] = m_simpleWeightsOrientations;
        state["complex_orientations"] = m_complexCellsOrientations;
        std::ofstream ofs(fileName + "orientations.json");
        if (ofs.is_open()) {
            ofs << std::setw(4) << state << std::endl;
        } else {
            std::cout << "cannot save neuron state statistics file" << std::endl;
        }
        ofs.close();
    }
}

/**
 * Resets the spike train of all neurons.
 */
void SpikingNetwork::resetSTrain() {
    size_t layer = 0;
    for (auto &neurons: m_neurons) {
            for (auto &neuron: neurons) {
                neuron.get().resetSpikeTrain();
            }
            ++layer;
    }
}

void SpikingNetwork::setEventsParameters(std::vector<std::vector<size_t>> parameters) {
    m_eventsParameters = parameters;
}

void SpikingNetwork::computeBinTS() {
    int layer = 0;
    std::vector<std::vector<double>> descriptor;
    int bin_start = 0;
    int bin_end = 1000000;
    int bin_add = bin_end;
    std::vector<std::vector<double>> binDescriptors(int((bin_end-bin_start)/bin_add));

    std::vector<size_t> function_output;
    bool to_save = false;
    int depth_simple = m_networkConf.getLayerConnectivity()[0].sizes[2];
    int depth_complex = m_networkConf.getLayerConnectivity()[1].sizes[2];
    float rem_val = 0;
    int layers[2] = {0, 1};
    for(int l: layers) {
        for(auto &neuron: m_neurons[l]) {
            int b = bin_start;
            int idex = 0;
            while(b < bin_end) {
                double desc = neuron.get().getTimeSurfaceBins(b,b + bin_add);
                binDescriptors[idex].push_back(desc);
                idex += 1;
                b += bin_add;
                rem_val += desc;
            }
        }
    }
    int b = bin_start;
    int idex = 0;
    while(b < bin_end) {
        descriptor.push_back(binDescriptors[idex]);
        idex += 1;
        b += bin_add;
    }

    to_save = true;
    auto is_null = [](double i){ return i != 0; };
    if(rem_val!=0) {
        for(auto param : m_eventsParameters[0]) {
            function_output.push_back(param);
        }
        if(to_save) {
            saveNewDescriptor(descriptor, function_output);
        }
    }

}

void SpikingNetwork::saveNewDescriptor(std::vector<std::vector<double>> x, std::vector<size_t> y) {
    int it = 0;
    fs::create_directory(m_networkConf.getNetworkPath() + "statistics/mnist" + "/");
    fs::create_directory(m_networkConf.getNetworkPath() + "statistics/mnist" + "/" + std::to_string(y[0]) + "/");
    for (const auto & file : fs::directory_iterator{m_networkConf.getNetworkPath() + "statistics/mnist" + "/"
                        + std::to_string(y[0]) + "/" }) 
        {
        it+=1;
    }
    nlohmann::json state;
    state["x"] = x;
    state["y"] = y;
    std::ofstream ofs(m_networkConf.getNetworkPath() + "statistics/mnist" +  "/" + 
                    std::to_string(y[0]) + "/" + std::to_string(it) + ".json");
    if (ofs.is_open()) {
        ofs << std::setw(4) << state << std::endl;
    } else {
        std::cout << "cannot save neuron state supervised dataset file" << std::endl;
    }
    ofs.close();
    it+=1;
}

void SpikingNetwork::setDynamicActivation(bool activation) {
    m_activation = activation;
}

bool SpikingNetwork::flushEvents(const Event &event) { // to be used for surround suppression
    int x_center = 183;
    int y_center = 133;
    int x_min = x_center - int(10 * m_patchSize);
    int x_max = x_center + int(10 * m_patchSize); 
    int y_min = y_center - int(10 * m_patchSize);
    int y_max = y_center + int(10 * m_patchSize);
    if(event.x() >= x_min && event.x() <= x_max && event.y() >= y_min && event.y() <= y_max) {
        return false;
    }
    return true;
}