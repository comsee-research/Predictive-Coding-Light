//
// Created by Thomas on 14/04/2021.
//

#ifndef NEUVISYS_DV_NEURON_HPP
#define NEUVISYS_DV_NEURON_HPP

#include <cmath>
#include <list>
#include <iomanip>
#include <utility>

#include <opencv2/core/mat.hpp>

#include "network/config/Config.hpp"
#include "network/utils/Util.hpp"
#include "network/utils/WeightMap.hpp"
#include "network/utils/WeightMatrix.hpp"

/* Abstract class defining a Neuron.
 */
class Neuron {
protected:
    size_t m_index;
    size_t m_layer;
    NeuronConfig m_conf; 
    double m_eLTP;
    double m_eLTD;
    double m_iLTP;
    double m_iLTD;
    Position m_pos{};
    Position m_offset{};
    WeightMap m_weights;
    WeightMap m_topDownInhibitionWeights;
    WeightMap m_lateralInhibitionWeights;
    WeightMap m_topDownExcitationWeights;
    WeightMap m_lateralExcitationWeights;
    boost::circular_buffer<NeuronEvent> m_lateralLocalInhibitionEvents;
    std::vector<std::reference_wrapper<Neuron>> m_outConnections;
    std::vector<std::reference_wrapper<Neuron>> m_inConnections;
    std::vector<std::reference_wrapper<Neuron>> m_lateralLocalInhibitionConnections;
    std::vector<std::reference_wrapper<Neuron>> m_topDownDynamicInhibitionConnections;
    std::vector<std::reference_wrapper<Neuron>> m_lateralDynamicInhibitionConnections;
    std::vector<size_t> m_range;
    size_t m_spikingTime{};
    size_t m_lastSpikingTime{};
    size_t m_totalSpike{};
    size_t m_spikeRateCounter{};
    size_t m_activityCounter{};
    double m_decay;
    double m_potential{};
    double m_adaptationPotential{};
    double m_threshold;
    size_t m_timestampLastEvent{};
    bool m_spike;
    bool m_noLat{false};
    size_t m_lifeSpan{};
    double m_spikingRateAverage{};
    std::vector<size_t> m_trackingSpikeTrain;
    std::vector<std::pair<double, uint64_t>> m_trackingPotentialTrain;
    std::vector<double> m_potentialThreshold;
    std::vector<size_t> m_amount_of_events;
    std::vector<std::vector<std::vector<double>>> m_sumOfInhibWeights;
    std::vector<std::vector<double>> m_sumOfTopDownWeights;
    std::vector<std::vector<std::vector<double>>> m_sumOfExcitWeights;
    std::vector<std::vector<double>> m_sumOfTopDownExcitWeights;
    std::vector<int> m_start;
    std::vector<std::tuple<double, uint64_t>> m_excitatoryEvents;
    std::vector<std::vector<std::tuple<double, double, uint64_t>>> m_timingOfInhibition;
    double m_spikingPotential{};
    double m_beforeInhibitionPotential{};
    int m_negativeLimits;
    size_t m_firstInput{};
    std::vector<double> m_outputs;
    int m_spikeCounter;

    void writeJson(nlohmann::json &state);

    void readJson(const nlohmann::json &state);

    virtual void potentialDecay(size_t time);

    virtual double refractoryPotential(size_t time);

    virtual void adaptationPotentialDecay(size_t time);

    virtual void spikeRateAdaptation();

    void checkNegativeLimits();

    void setLastBeforeInhibitionPotential();

private:
    virtual void spike(size_t /* time */) {};

public:
    Neuron(size_t index, size_t layer, NeuronConfig conf, Position pos, Position offset);

    [[nodiscard]] virtual size_t getIndex() const { return m_index; }

    [[nodiscard]] virtual size_t getLayer() const { return m_layer; }

    [[nodiscard]] virtual Position getPos() const { return m_pos; }

    [[nodiscard]] virtual Position getOffset() const { return m_offset; }

    [[nodiscard]] virtual double getThreshold() const { return m_threshold; }

    [[nodiscard]] virtual double getSpikingRate() const { return m_spikingRateAverage; }

    [[nodiscard]] virtual size_t getSpikingTime() const { return m_spikingTime; }

    [[nodiscard]] virtual double getSpikingPotential() const { return m_spikingPotential; }

    [[nodiscard]] virtual double getDecay() const { return m_decay; }

    [[nodiscard]] virtual double getAdaptationPotential() const { return m_adaptationPotential; }

    [[nodiscard]] virtual size_t getActivityCount();

    virtual void resetActivityCount();

    [[nodiscard]] virtual NeuronConfig getConf() const { return m_conf; }

    virtual void setPotentialTrack(std::vector<int> val) { m_conf.POTENTIAL_TRACK = std::move(val);}

    virtual double getTopDownInhibitionWeights(size_t neuronId) { return m_topDownInhibitionWeights.at(neuronId); }

    virtual double getTopDownExcitationWeights(size_t neuronId) { return m_topDownExcitationWeights.at(neuronId); }

    virtual double getNormTopDownInhibitionWeights() { return m_topDownInhibitionWeights.getL1Norm(); }

    virtual double getNormTopDownExcitationWeights() { return m_topDownExcitationWeights.getL1Norm(); }

    virtual double getlateralInhibitionWeights(size_t neuronId) { return m_lateralInhibitionWeights.at(neuronId); }

    virtual double getlateralExcitationWeights(size_t neuronId) { return m_lateralExcitationWeights.at(neuronId); }

    virtual double getNormLateralInhibitionWeights() { return m_lateralInhibitionWeights.getL1Norm(); }

    virtual double getNormLateralExcitationWeights() { return m_lateralExcitationWeights.getL1Norm(); }

    virtual void normalizeL1Weights() {};

    [[nodiscard]] virtual WeightMap &getWeightsMap() { return m_weights; }

    [[nodiscard]] virtual double getWeightsMapNorm() { return m_weights.getNorm(); }

    virtual WeightMatrix &getWeightsMatrix() {};

    virtual double getWeightsMatrixNorm() {};

    virtual std::vector<size_t> getWeightsDimension() { return m_weights.getDimensions(); }

    virtual void weightUpdate() {};

    virtual cv::Mat summedWeightMatrix() { return {}; };

    virtual void resetSpike() { m_spike = false; }

    [[nodiscard]] virtual std::vector<std::reference_wrapper<Neuron>>
    getOutConnections() const { return m_outConnections; }

    [[nodiscard]] virtual std::vector<std::reference_wrapper<Neuron>>
    getInConnections() const { return m_inConnections; }

    [[nodiscard]] virtual std::vector<std::reference_wrapper<Neuron>>
    getLateralLocalInhibitionConnections() const { return m_lateralLocalInhibitionConnections; }

    [[nodiscard]] virtual std::vector<std::reference_wrapper<Neuron>>
    getTopDownDynamicInhibitionConnections() const { return m_topDownDynamicInhibitionConnections; }

    [[nodiscard]] virtual std::vector<std::reference_wrapper<Neuron>>
    getLateralDynamicInhibitionConnections() const { return m_lateralDynamicInhibitionConnections; }

    virtual const std::vector<size_t> &getTrackingSpikeTrain() { return m_trackingSpikeTrain; }

    virtual const std::vector<std::pair<double, size_t>> &getTrackingPotentialTrain() {return m_trackingPotentialTrain; }

    virtual double getPotential(size_t time);

    virtual size_t getLastSpikingTime() const { return m_lastSpikingTime; }

    virtual double getBeforeInhibitionPotential() { return m_beforeInhibitionPotential; }

    virtual void saveWeights(const std::string &filePath) {};

    virtual void savePlasticLocalInhibitionWeights(const std::string &filePath) {};

    virtual void saveState(std::string &filePath);

    virtual void loadState(std::string &filePath);

    virtual void loadWeights(std::string &filePath) {};

    virtual void loadWeights(cnpy::npz_t &arrayNPZ) {};

    virtual void loadPlasticLocalInhibitionWeights(std::string &filePath) {};

    virtual void loadPlasticLocalInhibitionWeights(cnpy::npz_t &arrayNPZ) {};

    virtual void loadLateralInhibitionWeights(cnpy::npz_t &arrayNPZ) {};

    virtual void loadTopDownInhibitionWeights(cnpy::npz_t &arrayNPZ) {};

    virtual void loadLateralInhibitionWeights(std::string &filePath) {};

    virtual void loadTopDownInhibitionWeights(std::string &filePath) {};

    virtual void saveTopDownInhibitionWeights(std::string &filePath) {};

    virtual void saveLateralInhibitionWeights(std::string &filePath) {};

    virtual void loadLateralExcitationWeights(cnpy::npz_t &arrayNPZ) {};

    virtual void loadTopDownExcitationWeights(cnpy::npz_t &arrayNPZ) {};

    virtual void loadLateralExcitationWeights(std::string &filePath) {};

    virtual void loadTopDownExcitationWeights(std::string &filePath) {};

    virtual void saveTopDownExcitationWeights(std::string &filePath) {};

    virtual void saveLateralExcitationWeights(std::string &filePath) {};   

    virtual void assignTonoLat(bool lat) {m_noLat = lat; };

    virtual void assignToPotentialTrain(std::pair<double, uint64_t> potential);

    virtual void assignToPotentialThreshold();

    virtual void assignToAmountOfEvents(int type);

    virtual void assignToSumLateralWeights(int type, Position pos, double wi, size_t depth);

    virtual void assignToSumLateralExcitatoryWeights(int type, Position pos, double wi, size_t depth);

    virtual void assignToSumTopDownWeights(int index, double wi, size_t depth);

    virtual void assignToSumTopDownExcitatoryWeights(int index, double wi, size_t depth);

    virtual void assignToTimingOfInhibition(int type, std::tuple<double, double, uint64_t> variation);

    virtual void assignToExcitatoryEvents(std::tuple<double, uint64_t> event);

    virtual std::vector<std::pair<double, uint64_t>> getPotentialTrain();

    virtual std::vector<double> getPotentialThreshold();

    virtual std::vector<size_t> getAmountOfEvents();

    virtual std::vector<std::vector<std::vector<double>>> getSumLateralWeights();

    virtual std::vector<std::vector<double>> getSumTopDownWeights();

    virtual std::vector<std::vector<std::vector<double>>> getSumLateralExcitatoryWeights();

    virtual std::vector<std::vector<double>> getSumTopDownExcitatoryWeights();

    virtual std::vector<std::vector<std::tuple<double, double, uint64_t>>> getTimingOfInhibition();

    virtual std::vector<std::tuple<double, uint64_t>> getExcitatoryEvents();

    virtual void thresholdAdaptation();

    virtual void addOutConnection(Neuron &neuron);

    virtual void addInConnection(Neuron &neuron);

    virtual void initInWeights(size_t id);

    virtual void addTopDownDynamicInhibitionConnection(Neuron &neuron);

    virtual void initTopDownDynamicInhibitionWeights(size_t id);

    virtual void initTopDownDynamicExcitationWeights(size_t id);

    virtual void addLateralLocalInhibitionConnections(Neuron &neuron);

    virtual void addLateralDynamicInhibitionConnections(Neuron &neuron);

    virtual void initLateralDynamicInhibitionWeights(size_t id);

    virtual void initLateralDynamicExcitationWeights(size_t id);

    virtual bool newEvent(Event event) {};

    virtual bool newEvent(NeuronEvent /* event */) { return false; };

    virtual void newLocalInhibitoryEvent(NeuronEvent /* event */) {};

    virtual void newTopDownInhibitoryEvent(NeuronEvent /* event */) {};

    virtual bool newTopDownExcitatoryEvent(NeuronEvent /* event */) {};

    virtual void newLateralInhibitoryEvent(NeuronEvent /* event */) {};

    virtual bool newLateralExcitatoryEvent(NeuronEvent /* event */) {};

    virtual void setNeuromodulator(double /* neuromodulator */) {};

    virtual void setInhibitionRange(std::vector<size_t> inhibitionRange);

    virtual void trackPotential(size_t time);

    virtual void updateState(size_t timeInterval);

    virtual double updateKernelSpikingRate(long /* time */) { return 1.; };

    virtual void learningDecay(double count);

    virtual void resetNeuron();

    virtual void randomInhibition(int norm_factor);

    virtual void shuffleLateralInhibition();

    virtual void shuffleTopDownInhibition();

    virtual void resetSpikeTrain();

    virtual double getTimeSurfaceBins(double t, double tf) {};

    int getOutputSize() {return m_outputs.size(); };

    virtual int getSpikeCount() {return m_spikeCounter;};

    double getLateralWeightsAverage();

    double getMinimumLateralWeight();

    double getMaximumLateralWeight();

};

#endif //NEUVISYS_DV_NEURON_HPP
