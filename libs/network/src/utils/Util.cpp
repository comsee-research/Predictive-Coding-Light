//
// Created by Thomas on 14/04/2021.
//

#include "utils/Util.hpp"



namespace Util {
    bool fileExist(std::string &filePath) {
        if (FILE *file = fopen(filePath.c_str(), "r")) {
            fclose(file);
            return true;
        } else {
            return false;
        }
    }

    bool endsWith(std::string const &value, std::string const &ending) {
        if (ending.size() > value.size()) return false;
        return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
    }

    void saveEventFile(std::vector<Event> &events, std::string &filePath) {
        std::vector<double> timestamp(events.size());
        std::vector<double> x(events.size());
        std::vector<double> y(events.size());
        std::vector<double> polarity(events.size());
        std::vector<double> camera(events.size());
        size_t count = 0;
        for (auto const &event: events) {
            timestamp[count] = static_cast<double>(event.timestamp());
            x[count] = event.x();
            y[count] = event.y();
            polarity[count] = event.polarity();
            camera[count] = event.camera();
            ++count;
        }
        cnpy::npz_save(filePath + ".npz", "arr_0", &timestamp[0], {events.size()}, "w");
        cnpy::npz_save(filePath + ".npz", "arr_1", &x[0], {events.size()}, "a");
        cnpy::npz_save(filePath + ".npz", "arr_2", &y[0], {events.size()}, "a");
        cnpy::npz_save(filePath + ".npz", "arr_3", &polarity[0], {events.size()}, "a");
        cnpy::npz_save(filePath + ".npz", "arr_4", &camera[0], {events.size()}, "a");
    }

    int winnerTakeAll(std::vector<size_t> vec) {
        std::vector<size_t> argsmax;
        size_t max = 0;

        for (size_t i = 0; i < vec.size(); ++i) {
            if (vec[i] > max) {
                max = vec[i];
                argsmax.clear();
            }

            if (max != 0 && vec[i] == max) {
                argsmax.push_back(i);
            }
        }

        if (argsmax.empty()) {
            return -1;
        } else {
            std::vector<int> randomArgmax;
            std::sample(argsmax.begin(), argsmax.end(), std::back_inserter(randomArgmax), 1, generator);
            return randomArgmax[0];
        }
    }

    double secondOrderNumericalDifferentiationMean(const std::vector<double> &vec, long n) {
        if (!vec.empty()) {
            double sumDVec = 0;
            int count = 0;
            for (size_t i = vec.size() - 2; i > vec.size() - n && i > 0; --i) {
                sumDVec += (vec[i + 1] - vec[i - 1]) / 2.0;
                ++count;
            }
            return sumDVec / count;
        } else {
            return 0;
        }
    }

    void ornsteinUhlenbeckProcess(double &pos, double dt, double theta, double mu, double sigma) {
        double noise = normalDistr(generator) * std::sqrt(dt);
        pos = theta * (mu - pos) * dt + sigma * noise;
    }
}
