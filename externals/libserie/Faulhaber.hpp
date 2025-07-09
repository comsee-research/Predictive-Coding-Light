#ifndef FAULHABER__HPP___
#define FAULHABER__HPP___

#include "comserie.hpp"

class Faulhaber : public ComSerie {
public:
    Faulhaber() = default;
    explicit Faulhaber(int Adresse_moteur = 0, const string& port = "/dev/ttyUSB0");

    ~Faulhaber();

    void StartDrive();

    void StopDrive();

    double GetPosition();

    void SetAbsolutePosition(int position);

    void SetRelativePosition(int position);

    void SetSpeed(int speed);

private:
    string Adresse_moteur_;
};

#endif
