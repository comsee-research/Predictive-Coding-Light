//
// Created by Thomas on 14/04/2021.
//

#ifndef NEUVISYS_DV_SIMPLENEURON_HPP
#define NEUVISYS_DV_SIMPLENEURON_HPP

#include "Neuron.hpp"
struct CompareEventsTimestamp {
    bool operator()(Event const &event1, Event const &event2) {
        return event1.timestamp() > event2.timestamp();
    }
};

class SimpleNeuron : public Neuron {
protected:
    std::vector<size_t> m_delays;
    boost::circular_buffer<Event> m_events;
    boost::circular_buffer<NeuronEvent> m_topDownInhibitionEvents;
    boost::circular_buffer<NeuronEvent> m_lateralInhibitionEvents;
    boost::circular_buffer<NeuronEvent> m_topDownExcitationEvents;
    boost::circular_buffer<NeuronEvent> m_lateralExcitationEvents;
    std::priority_queue<Event, std::vector<Event>, CompareEventsTimestamp> m_waitingList;
    WeightMatrix &m_sharedWeights;
    WeightMatrix &m_sharedInhibWeights;
    double &m_outsideLatNorm;
    int m_sizeInit;
    double m_pliLTP;
    double m_pliLTD;
    double m_ptiLTP;
    double m_ptiLTD;

public:
    SimpleNeuron(size_t index, size_t layer, NeuronConfig &conf, Position pos, Position offset, WeightMatrix &weights, WeightMatrix &localInhibWeights, size_t nbSynapses, double &outsideLat);

    bool newEvent(Event event) override;

    void newLocalInhibitoryEvent(NeuronEvent event) override;

    void newTopDownInhibitoryEvent(NeuronEvent event) override;

    void newLateralInhibitoryEvent(NeuronEvent event) override;

    bool newTopDownExcitatoryEvent(NeuronEvent event) override;

    bool newLateralExcitatoryEvent(NeuronEvent event) override;

    void normalizeL1Weights() override;

    WeightMatrix &getWeightsMatrix() override;

    double getWeightsMatrixNorm() override;

    std::vector<size_t> getWeightsDimension() override;

    void saveWeights(const std::string &filePath) override;

    void savePlasticLocalInhibitionWeights(const std::string &filePath) override;

    void saveLateralInhibitionWeights(std::string &filePath) override;

    void saveTopDownInhibitionWeights(std::string &filePath) override;

    void saveLateralExcitationWeights(std::string &filePath) override;

    void saveTopDownExcitationWeights(std::string &filePath) override;

    void loadWeights(cnpy::npz_t &arrayNPZ) override;

    void loadWeights(std::string &filePath) override;

    void loadPlasticLocalInhibitionWeights(std::string &filePath) override;

    void loadPlasticLocalInhibitionWeights(cnpy::npz_t &arrayNPZ) override;

    void loadLateralInhibitionWeights(cnpy::npz_t &arrayNPZ) override;

    void loadLateralInhibitionWeights(std::string &filePath) override;

    void loadTopDownInhibitionWeights(cnpy::npz_t &arrayNPZ) override;

    void loadTopDownInhibitionWeights(std::string &filePath) override;

    void loadLateralExcitationWeights(cnpy::npz_t &arrayNPZ) override;

    void loadLateralExcitationWeights(std::string &filePath) override;

    void loadTopDownExcitationWeights(cnpy::npz_t &arrayNPZ) override;

    void loadTopDownExcitationWeights(std::string &filePath) override;

    bool checkRemainingEvents(size_t time) { return !m_waitingList.empty() && m_waitingList.top().timestamp() <= time; }

    void weightUpdate() override;

    double getTimeSurfaceBins(double t, double tf);

    void resetSimpleNeuron();

private:
    bool membraneUpdate(Event event);

    void spike(size_t time) override;

    void updateTimeSurface(double time, bool polarity);
};

#endif //NEUVISYS_DV_SIMPLENEURON_HPP
