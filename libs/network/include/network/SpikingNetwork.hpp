//
// Created by Thomas on 14/04/2021.
//

#ifndef NEUVISYS_DV_SPIKING_NETWORK_HPP
#define NEUVISYS_DV_SPIKING_NETWORK_HPP

#include <utility>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

#include "neurons/SimpleNeuron.hpp"
#include "neurons/ComplexNeuron.hpp"

class SpikingNetwork {
    NetworkConfig m_networkConf;
    NeuronConfig m_simpleNeuronConf;
    NeuronConfig m_complexNeuronConf;

    std::vector<WeightMatrix> m_sharedWeightsSimple;
    std::vector<WeightMatrix> m_sharedInhibWeightsSimple;
    std::vector<std::vector<WeightMatrix>> m_sharedWeightsComplex;
    std::vector<std::vector<WeightMatrix>> m_sharedInhibWeightsComplex;

    std::priority_queue<Event, std::vector<Event>, CompareEventsTimestamp> m_eventsList;
    long m_lastEventTs;

    std::vector<std::map<std::tuple<uint64_t, uint64_t, uint64_t>, uint64_t>> m_layout;
    std::vector<size_t> m_structure;
    std::vector<std::vector<std::reference_wrapper<Neuron>>> m_neurons;

    std::vector<SimpleNeuron> m_simpleNeurons;
    std::vector<std::vector<ComplexNeuron>> m_complexNeurons;
    std::vector<std::vector<uint64_t>> m_pixelMapping;

    double m_neuromodulator{};
    std::vector<std::vector<std::vector<int>>> m_simpleWeightsOrientations;
    std::vector<std::vector<std::vector<int>>> m_complexCellsOrientations;
    double m_averageActivity{};

    std::vector<std::vector<size_t>> m_eventsParameters;

    bool m_activation;
    
    double m_patchSize{};
    double m_LatNorm{};


public:
    SpikingNetwork();

    explicit SpikingNetwork(const std::string &networkPath);

    void addLayer(const std::string &sharingType, const LayerConnectivity &connections);

    void addEvent(const Event &event);

    void updateNeuronsStates(long timeInterval);

    void loadWeights();

    void saveNetwork();

    void transmitNeuromodulator(double neuromodulator);

    void normalizeWeights();

    [[nodiscard]] double getAverageActivity() const { return m_averageActivity; }

    std::reference_wrapper<Neuron> &getNeuron(size_t index, size_t layer);

    const std::vector<size_t> &getNetworkStructure() { return m_structure; }

    std::vector<std::map<std::tuple<uint64_t, uint64_t, uint64_t>, uint64_t>> &getLayout() { return m_layout; }

    void intermediateSave(size_t saveCount);

    void saveStatistics(int simulation, int sequence, const std::string& folderName, bool sep_speed = false, int max_speed = 3);

    void changeTrack(int n_x, int n_y);

    void randomLateralInhibition(int norm_factor);

    void shuffleInhibition(int cases);

    void assignOrientations(int index_z, int orientation, int thickness);

    void assignComplexOrientations(int id, int orientation, int thickness);

    void saveOrientations();

    void resetSTrain();

    void processSynapticEvent();

    void setEventsParameters(std::vector<std::vector<size_t>> parameters);

    size_t getEventsParameters() {return m_eventsParameters.size();};

    void resetAllNeurons();

    void setDynamicActivation(bool activation);

    void assignPatchSize(double patch) {m_patchSize = patch;}


private:
    void saveNeuronsStates();

    static void neuronsStatistics(uint64_t time, int type_, Position pos, Neuron &neuron, double wi, bool spike=false, bool recordAllPotentials = false);

    void saveStatesStatistics(std::string &fileName, Neuron &neuron);

    void writeJsonNeuronsStatistics(nlohmann::json &state, Neuron &neuron);

    void generateWeightSharing(const LayerConnectivity &connections, size_t nbNeurons);

    void addNeuronEvent(const Neuron &neuron);

    void connectLayer(const LayerConnectivity &connections);

    static void topDownDynamicInhibition(Neuron &neuron);

    void topDownDynamicExcitation(Neuron &neuron);

    static void lateralLocalInhibition(Neuron &neuron);

    void lateralDynamicInhibition(Neuron &neuron);

    void lateralDynamicExcitation(Neuron &neuron);

    void topDownConnection(Neuron &neuron, const std::vector<int> &interConnections, const size_t currLayer, const std::vector<std::vector<size_t>> &neuronSizes, const
    std::vector<std::string> &inhibition);

    void lateralLocalInhibitionConnection(Neuron &neuron, size_t currLayer, const std::vector<size_t> &layerSizes);

    void lateralDynamicInhibitionConnection(Neuron &neuron, size_t currLayer,
                                            const std::vector<std::vector<size_t>> &layerPatches,
                                            const std::vector<size_t> &layerSizes);

    void saveNetworkLayout();

    void computeBinTS();

    void saveNewDescriptor(std::vector<std::vector<double>> x, std::vector<size_t> y);

    bool flushEvents(const Event &event);

};

#endif //NEUVISYS_DV_SPIKING_NETWORK_HPP
