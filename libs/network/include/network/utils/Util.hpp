//
// Created by Thomas on 14/04/2021.
//

#ifndef NEUVISYS_UTIL_HPP
#define NEUVISYS_UTIL_HPP

#include <random>
#include <chrono>

#include "cnpy/cnpy.h"

#include "Types.hpp"

#define UNUSED(expr) (void)(expr)

static constexpr std::size_t NBPOLARITY = 2;
static std::mt19937 generator(static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count()));
static std::normal_distribution<double> normalDistr(0.0, 1.0);
static std::uniform_real_distribution<double> uniformRealDistr(0.0, 1.0);
namespace Util {

    int winnerTakeAll(std::vector<size_t> vec);

    double secondOrderNumericalDifferentiationMean(const std::vector<double> &vec, long n);

    bool fileExist(std::string &filePath);

    bool endsWith(std::string const &value, std::string const &ending);

    void ornsteinUhlenbeckProcess(double &pos, double dt, double theta, double mu, double sigma);

    void saveEventFile(std::vector<Event> &events, std::string &filePath);
}

class Position {
    uint64_t m_posx{};
    uint64_t m_posy{};
    uint64_t m_posz{};

public:
    inline Position() : m_posx(0), m_posy(0), m_posz(0) {};

    inline Position(uint64_t x, uint64_t y, uint64_t z) : m_posx(x), m_posy(y), m_posz(z) {}

    inline Position(uint64_t x, uint64_t y) : m_posx(x), m_posy(y), m_posz(0) {}

    [[nodiscard]] inline uint64_t x() const { return m_posx; }

    [[nodiscard]] inline uint64_t y() const { return m_posy; }

    [[nodiscard]] inline uint64_t z() const { return m_posz; }
};

#endif //NEUVISYS_UTIL_HPP
