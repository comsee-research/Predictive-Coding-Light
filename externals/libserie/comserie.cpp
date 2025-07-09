/**
 * \file comserie.cpp
 * \author S�bastien.BONNET
 * \date 31/10/05
 **/

#include "comserie.hpp"

//Declaration des static
int ComSerie::nb_instance_;
struct termios ComSerie::old_;

//==============================================================================

ComSerie::ComSerie() {
    if (nb_instance_ == 0) comOuverte_ = false;
}
//==============================================================================

ComSerie::~ComSerie() {
}
//==============================================================================

int ComSerie::ouvrir(string port, unsigned int parametresCommunication, unsigned int timeout) {
    port_ = port;
    parametresCommunication_ = parametresCommunication;
    timeout_ = timeout;

    if (nb_instance_ == 0) {
        fd_ = open(port_.c_str(), O_RDWR | O_NOCTTY); //ouverture du port com
        if (fd_ < 0) {
            perror(port_.c_str());
            return -1;
        }
        comOuverte_ = true;

        tcgetattr(fd_, &old_); //save current serial port settings

        //configuration du port s�rie
        config_();
    }
    nb_instance_++;
    std::cout << " nb instance : " << nb_instance_ << std::endl;
    return 0;
}
//==============================================================================

void ComSerie::config_() {
    bzero(&tios_, sizeof(tios_));

    tios_.c_cflag = parametresCommunication_;
    tios_.c_iflag = 0;
    tios_.c_oflag = 0;
    tios_.c_lflag = 0;
    tios_.c_cc[VTIME] = timeout_;
    tios_.c_cc[VMIN] = 0;

    tcflush(fd_, TCIFLUSH);
    tcsetattr(fd_, TCSANOW, &tios_);
}
//==============================================================================

int ComSerie::fermer() {
    nb_instance_--;
    if (nb_instance_ == 0) {
        if (comOuverte_) {
            //tcsetattr(fd_,TCSANOW,&old_); //restore the old port settings
            close(fd_);
            comOuverte_ = false;
            return 0;
        } else return -1;
    } else {

        return 0;
    }
}
//==============================================================================

void ComSerie::flush() {
    if (comOuverte_) tcflush(fd_, TCIFLUSH);
}
//==============================================================================

int ComSerie::envoyer(const uint8_t *buf, const uint32_t taille) {
    return write(fd_, buf, taille);
}
//==============================================================================

int ComSerie::recevoir(uint8_t *buf, const uint32_t tailleVoulue) {
    uint32_t nbRecv = 0;
    uint32_t ret;
    do {
        ret = read(fd_, buf + nbRecv, (tailleVoulue - nbRecv) * sizeof(uint8_t));
        nbRecv += ret;
        //  std::cout << " ret : " << ret << " nbRecv : " << nbRecv<< std::endl;
    } while (ret && nbRecv != tailleVoulue);

    if (!ret) return 0;
    return nbRecv;
}
//==============================================================================
