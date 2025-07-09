//
// Created by thomas on 06/07/22.
//

#include "utils/WeightMatrix.hpp"

std::random_device WeightMatrix::m_rd;
std::mt19937 WeightMatrix::m_generator(WeightMatrix::m_rd());
std::uniform_real_distribution<double> WeightMatrix::m_uniformRealDistr(0.0, 1.0);
std::uniform_real_distribution<double> WeightMatrix::m_uniformRealDistrConst(15.0, 15.0);


WeightMatrix::WeightMatrix() : m_data() {
    m_dimensions.push_back(0);
}

WeightMatrix::WeightMatrix(std::vector<size_t> dimensions) : m_data(), m_dimensions(std::move(dimensions)) {

}

WeightMatrix::WeightMatrix(std::vector<size_t> dimensions, bool uniform, double normFactor) : m_dimensions(std::move(dimensions)) {
    size_t nbWeights = 1;
    for (const auto d : m_dimensions) {
        nbWeights *= d;
    }
    m_data.reserve(nbWeights);

    for (size_t i = 0; i < nbWeights; ++i) {
        if (uniform) {
            m_data.emplace_back(m_uniformRealDistr(m_generator));
        } else {
            m_data.emplace_back(m_uniformRealDistrConst(m_generator));//0);
        }
    }
    // normalize(normFactor);
}

void WeightMatrix::operator=(WeightMatrix weight) {
    size_t nbWeights = 1;
    for(auto d: weight.getDimensions()) {
        m_dimensions.push_back(d);
        nbWeights *= d;
    }
    m_data.reserve(nbWeights);

    for (auto wi: weight.getMatrix()) {
        m_data.emplace_back(wi);
    }
}

double WeightMatrix::getNorm() {
    double norm = 0;
    for (const auto &element : m_data) {
        norm += pow(element, 2);
    }
    return sqrt(norm);
}

void WeightMatrix::antiNoise() {
    // double norm = getNorm();
    double med = getMedian();
    double med_thresh = 2;
    double threshold = 0.25;
    for (auto &element : m_data) {
        if(element > med_thresh) {
            element = med;
        }
    }
}

double WeightMatrix::getL1Norm() {
    double norm = 0;
    for (const auto &element : m_data) {
        norm += std::fabs(element);
    }
    return norm;
}

void WeightMatrix::normalize(const double normFactor) {
    auto norm = getNorm();
    if (norm != 0) {
        for (auto &element : m_data) {
            element *= normFactor / norm;
        }
    }
}

void WeightMatrix::MinMaxNorm(const double normFactor) {
    
    auto max = getMaximum();
    auto min = getMinimum();
    if ((max-min) !=0) {
        for (auto &element : m_data) {
            element = (element - min) / (max - min);
        }
    }
    auto norm = getNorm();
    if (norm !=0) {
        for (auto &element : m_data) {
            element *= normFactor / norm;
        }
    }
}

void WeightMatrix::pruning(double thresh) {
    for (auto &element : m_data) {
        if(element < thresh) {
            element = 0;
        }
    }
}

double WeightMatrix::getMaximum() {
    double maximum = 0;
    for (const auto &element : m_data) {
        if(element > maximum) {
            maximum = std::abs(element);
        } 
    }
    return maximum;
}

double WeightMatrix::getMinimum() {
    double minimum = 50000;
    for (const auto &element : m_data) {
        if(element < minimum) {
            minimum = std::abs(element);
        } 
    }
    return minimum;
}

double WeightMatrix::getMedian() {
    std::vector<double> weightLoc;
    for (const auto &element : m_data) {
        weightLoc.push_back(-element);
    }
    std::sort(weightLoc.begin(), weightLoc.end());
    return weightLoc[int(round(weightLoc.size()/2) + round(weightLoc.size()/4))];
}

int WeightMatrix::getNMaximum(int x, int y, int n) {
    std::vector<std::pair<double,int>> weightLoc;
    for(int i=0; i < m_dimensions[2]; i++) {
        weightLoc.push_back(std::make_pair(-this->get(x,y,i), i));
    }
    std::sort(weightLoc.begin(), weightLoc.end());
    return weightLoc[n].second;
}

void WeightMatrix::normalizeL1Norm(const double normFactor) {
    auto norm = getL1Norm();
    if (norm != 0) {
        for (auto &element : m_data) {
            element *= normFactor / norm;
        }
    }
}

void WeightMatrix::loadNumpyFile(const std::string &filePath) {
    // auto *weights = cnpy::npy_load(filePath).data<double>();
    cnpy::NpyArray arr = cnpy::npy_load(filePath);
    double *weights = arr.data<double>();
    weightsToMap(m_data, weights);
}

void WeightMatrix::loadNumpyFile(cnpy::npz_t &arrayNPZ, const std::string &arrayName) {
    auto *weights = arrayNPZ[arrayName].data<double>();
    weightsToMap(m_data, weights);
}

void WeightMatrix::saveWeightsToNumpyFile(const std::string &filePath) {
    std::vector<double> data(static_cast<size_t>(m_data.size()));
    mapToWeights(m_data, data);
    cnpy::npy_save(filePath + ".npy", &data[0], m_dimensions, "w");
}

void WeightMatrix::saveWeightsToNumpyFile(const std::string &filePath, const std::string &arrayName) {
    std::vector<double> data(static_cast<size_t>(m_data.size()));
    mapToWeights(m_data, data);
    cnpy::npz_save(filePath, arrayName, &data[0], m_dimensions, "a");
}

void WeightMatrix::mapToWeights(const std::vector<double> &map, std::vector<double> &data) {
    size_t count = 0;
    for (auto const &element: map) {
        data[count] = static_cast<double>(element);
        ++count;
    }
}

void WeightMatrix::weightsToMap(std::vector<double> &map, const double *weights) {
    size_t count = 0; 
    for (auto &element: map) {
        element = (weights[count]);
        ++count;
    }
}

double &WeightMatrix::get(size_t a) { // dimensional indexing
    return m_data.at(a);
}

double &WeightMatrix::get(size_t a, size_t b) {
    return m_data.at(b + a * m_dimensions[1]);
}

double &WeightMatrix::get(size_t a, size_t b, size_t c) {
    return m_data.at(c + b * m_dimensions[2] + a * m_dimensions[2] * m_dimensions[1]);
}

double &WeightMatrix::get(size_t a, size_t b, size_t c, size_t d) {
    return m_data.at(d + c * m_dimensions[3] + b * m_dimensions[3] * m_dimensions[2] + a * m_dimensions[3] * m_dimensions[2] * m_dimensions[1]);
}

double &WeightMatrix::get(size_t a, size_t b, size_t c, size_t d, size_t e) {
    return m_data.at(e + d * m_dimensions[4] + c * m_dimensions[4] * m_dimensions[3] + b * m_dimensions[4] * m_dimensions[3] * m_dimensions[2] + a * m_dimensions[4] *
    m_dimensions[3] * m_dimensions[2] * m_dimensions[1]);
}

void WeightMatrix::setSeed(size_t seed) {
    m_generator.seed(seed);
}
