//
// Created by Thomas on 14/04/2021.
//

#ifndef NEUVISYS_DV_COMPLEXNEURON_HPP
#define NEUVISYS_DV_COMPLEXNEURON_HPP

#include "Neuron.hpp"

class ComplexNeuron : public Neuron {
protected:
    boost::circular_buffer<NeuronEvent> m_events;
    std::vector<size_t> m_vectorSpikingTime;
    WeightMatrix &m_sharedWeights;
    WeightMatrix &m_sharedInhibWeights;

public:

    ComplexNeuron(size_t index, size_t layer, NeuronConfig &conf, Position pos, Position offset, 
                  const std::vector<size_t> &dimensions, const std::vector<size_t> &depth, WeightMatrix &weights, 
                  WeightMatrix &localInhibWeights);    

    bool newEvent(NeuronEvent event) override;

    void newLocalInhibitoryEvent(NeuronEvent event) override;

    void normalizeL1Weights() override;

    void saveWeights(const std::string &filePath) override;

    void loadWeights(std::string &filePath) override;

    void loadWeights(cnpy::npz_t &arrayNPZ) override;

    void weightUpdate() override;

    cv::Mat summedWeightMatrix() override;

    WeightMatrix &getWeightsMatrix() override;

    double getWeightsMatrixNorm() override;

    std::vector<size_t> getWeightsDimension() override;

    virtual double getTimeSurfaceBins(double t, double tf);

    void resetComplexNeuron();

private:
    bool membraneUpdate(NeuronEvent event);

    void spike(size_t time) override;

    void updateTimeSurface(double time);


};

#endif //NEUVISYS_DV_COMPLEXNEURON_HPP
