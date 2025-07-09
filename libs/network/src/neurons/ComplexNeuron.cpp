//
// Created by Thomas on 14/04/2021.
//

#include "ComplexNeuron.hpp"

/**
 *
 * @param index
 * @param layer
 * @param conf
 * @param pos
 * @param offset
 * @param dimensions
 */

ComplexNeuron::ComplexNeuron(size_t index, size_t layer, NeuronConfig &conf, Position pos, Position offset, const std::vector<size_t> &dimensions, const std::vector<size_t> &depth, WeightMatrix &weights, WeightMatrix &localInhibWeights) :
        Neuron(index, layer, conf, pos, offset),
        m_sharedWeights(weights),
        m_sharedInhibWeights(localInhibWeights),
        m_events(boost::circular_buffer<NeuronEvent>(1000)){
        m_weights = WeightMap(dimensions);
}

/**
 *
 * @param event
 * @return
 */
inline bool ComplexNeuron::newEvent(NeuronEvent event) {
    m_events.push_back(event);
    return membraneUpdate(event);
}

/**
 *
 * @param event
 */
void ComplexNeuron::newLocalInhibitoryEvent(NeuronEvent event) {
    m_lateralLocalInhibitionEvents.push_back(event);
    potentialDecay(event.timestamp());
    setLastBeforeInhibitionPotential();
    m_potential -=  (m_sharedInhibWeights.get(event.id()) + refractoryPotential(event.timestamp())); 
    checkNegativeLimits();
    m_timestampLastEvent = event.timestamp();
}

/**
 *
 * @param event
 * @return
 */
inline bool ComplexNeuron::membraneUpdate(NeuronEvent event) {
    potentialDecay(event.timestamp());
    if(m_weights.getSize()!=0) {
        m_potential += m_weights.at(event.id())
                    - refractoryPotential(event.timestamp());
    }
    else {
        m_potential += m_sharedWeights.get(event.x(), event.y(), event.z())
                - refractoryPotential(event.timestamp());
    }
    m_timestampLastEvent = event.timestamp();
    checkNegativeLimits();
    if (m_potential > m_threshold) {
        spike(event.timestamp());
        updateTimeSurface(event.timestamp());
        return true;
    }
    return false;
}

/**
 *
 * @param time
 */
inline void ComplexNeuron::spike(const size_t time) {
    m_vectorSpikingTime.push_back(time);
    m_lastSpikingTime = m_spikingTime;
    m_spikingTime = time;
    m_spike = true;
    ++m_spikeRateCounter;
    ++m_totalSpike;
    m_spikingPotential = m_potential;
    m_potential = m_conf.VRESET;
    // spikeRateAdaptation();
    if (m_conf.TRACKING == "partial") {
        m_trackingSpikeTrain.push_back(time - m_firstInput); // to find preferred orientations
    }
}

/**
 *
 */
inline void ComplexNeuron::weightUpdate() {
    if (m_conf.STDP_LEARNING == "excitatory" || m_conf.STDP_LEARNING == "all") {
        if(m_spikeRateCounter == 1) {

            double max_eWeight = 4; 
            double factor_eBound = 0.25; 
           for (NeuronEvent &event: m_events) {
                for(auto &t: m_vectorSpikingTime) {
                    if(int(t - event.timestamp()) >= 0) {
                        if(m_weights.getSize()!=0) {
                            m_weights.at(event.id()) += m_conf.ETA_LTP * exp(-static_cast<double>(std::fabs(int(t - event.timestamp()))) / m_conf.TAU_LTP);
                        }
                        else {
                            double wj = m_sharedWeights.get(event.x(), event.y(), event.z());
                            double softboundLTP =  (max_eWeight - wj) * factor_eBound; 
                            double dLTP = softboundLTP * m_eLTP * exp(-static_cast<double>(std::fabs(int(t - event.timestamp()))) / m_conf.TAU_LTP);
                            m_sharedWeights.get(event.x(), event.y(), event.z()) += dLTP;
                        }
                    }
                    else {
                        if(m_weights.getSize()!=0) {
                            m_weights.at(event.id()) += m_conf.ETA_LTD * exp(-static_cast<double>(std::fabs(int(t - event.timestamp()))) / m_conf.TAU_LTD);
                        }
                        else {
                            double wj = m_sharedWeights.get(event.x(), event.y(), event.z());
                            double softboundLTD =  (wj) * factor_eBound; 
                            double dLTD = softboundLTD * m_eLTD * exp(-static_cast<double>(std::fabs(int(t - event.timestamp()))) / m_conf.TAU_LTD);            
                            m_sharedWeights.get(event.x(), event.y(), event.z()) += dLTD;   
                            
                        }
                    }
                }
                if(m_weights.getSize()!=0) {
                    if (m_weights.at(event.id()) < 0) {
                        m_weights.at(event.id()) = 0;
                    } 
                }
                else{
                    if(m_sharedWeights.get(event.x(), event.y(), event.z()) < 0) {
                        m_sharedWeights.get(event.x(), event.y(), event.z()) = 0;
                    }
                }
                   
            }
            m_sharedWeights.normalizeL1Norm(m_conf.NORM_FACTOR);
            m_spikeRateCounter = 0;
            m_vectorSpikingTime.erase(m_vectorSpikingTime.begin(),m_vectorSpikingTime.begin()+m_vectorSpikingTime.size()-1);
        }

        double max_iWeight = 25;
        double factor_iBound = 0.04;
        for (NeuronEvent &event: m_lateralLocalInhibitionEvents) {
                double wj = m_sharedInhibWeights.get(event.id());
                double softboundLTP = (max_iWeight - wj) * factor_iBound; 
                double dLTP = softboundLTP * m_iLTP * exp(-static_cast<double>(m_spikingTime - event.timestamp()) / m_conf.TAU_LTP); // * 5 and same scale as stdp
                m_sharedInhibWeights.get(event.id()) += dLTP;
            if(m_lastSpikingTime != 0) {
                    double wj = m_sharedInhibWeights.get(event.id());
                    double softboundLTD = (wj) * factor_iBound; 
                    double dLTD = softboundLTD * m_iLTD * exp(-static_cast<double>(event.timestamp() - m_lastSpikingTime) / m_conf.TAU_LTD); // * 5
                    m_sharedInhibWeights.get(event.id()) += dLTD;
                
            }
            
            else {
                if (m_sharedInhibWeights.get(event.id()) < 0) {
                    m_sharedInhibWeights.get(event.id()) = 0;
                } 
            }
             
        }
        m_sharedInhibWeights.normalizeL1Norm(m_conf.ETA_INH);
    }
    m_events.clear();
    m_lateralLocalInhibitionEvents.clear();
}

void ComplexNeuron::resetComplexNeuron() {
    m_events.clear();
    m_lateralLocalInhibitionEvents.clear();
    m_vectorSpikingTime.clear();
}

void ComplexNeuron::normalizeL1Weights() {
    m_sharedWeights.normalizeL1Norm(m_conf.NORM_FACTOR);
    m_sharedInhibWeights.normalizeL1Norm(m_conf.ETA_INH);
}

/**
 *
 * @return
 */
WeightMatrix &ComplexNeuron::getWeightsMatrix() {
    return m_sharedWeights;
}

double ComplexNeuron::getWeightsMatrixNorm() {
    return m_sharedWeights.getNorm();
}

/**
 *
 * @return
 */
std::vector<size_t> ComplexNeuron::getWeightsDimension() {
    return m_sharedWeights.getDimensions();
}

/**
 *
 * @return
 */
inline cv::Mat ComplexNeuron::summedWeightMatrix() {
    auto dim = m_weights.getDimensions();

    cv::Mat mat = cv::Mat::zeros(static_cast<int>(dim[1]), static_cast<int>(dim[0]), CV_8UC3);
    double sum = 0, max = 0;
    for (int i = 0; i < dim[0]; ++i) {
        for (int j = 0; j < dim[1]; ++j) {
            for (int k = 0; k < dim[2]; ++k) {
                sum += m_weights.at(k + j * dim[2] + i * dim[2] * dim[1]);
            }
            if (sum > max) {
                max = sum;
            }
            auto &color = mat.at<cv::Vec3b>(j, i);
            color[0] = static_cast<unsigned char>(sum);
            sum = 0;
        }
    }
    mat = mat * 255.0 / max;
    return mat;
}

/**
 *
 * @param filePath
 */
void ComplexNeuron::saveWeights(const std::string &filePath) {
    auto weightsFile = filePath + std::to_string(m_index);
    auto weightsFileInhib = filePath + std::to_string(m_index) + "lli";
    m_sharedWeights.saveWeightsToNumpyFile(weightsFile);
    m_sharedInhibWeights.saveWeightsToNumpyFile(weightsFileInhib);
}

/**
 *
 * @param filePath
 */
void ComplexNeuron::loadWeights(std::string &filePath) {
    auto numpyFile = filePath + std::to_string(m_index) + ".npy";
    auto numpyFileInhib = filePath + std::to_string(m_index) + "lli.npy";
    m_sharedWeights.loadNumpyFile(numpyFile);
    m_sharedInhibWeights.loadNumpyFile(numpyFileInhib);
}

/**
 *
 * @param arrayNPZ
 */
void ComplexNeuron::loadWeights(cnpy::npz_t &arrayNPZ) {
    auto arrayName = std::to_string(m_index);
    if(m_weights.getSize()!=0) {
        m_weights.loadNumpyFile(arrayNPZ, arrayName);
    }
    else {
        m_sharedWeights.loadNumpyFile(arrayNPZ, arrayName);
    }
}

void ComplexNeuron::updateTimeSurface(double time) {
    m_outputs.push_back(time);
}

double ComplexNeuron::getTimeSurfaceBins(double t, double tf) {
    float timeSurfaceLocal = 0;
    for (auto & ts : m_outputs) {
        if(ts >= t && ts < tf) {
            timeSurfaceLocal+=1;
        }
    }
    return timeSurfaceLocal;
}