/**
 * \file comserie.hpp
 * \author S�bastien.BONNET
 * \date 31/10/05
 **/

#ifndef __COMSERIE_HPP__
#define __COMSERIE_HPP__

#include <iostream>
#include <stdint.h>
#include <termios.h>
#include <memory.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <string>

#define COM1 "/dev/ttyS0"
#define COM2 "/dev/ttyS1"
#define COM3 "/dev/ttyS2"
#define COM4 "/dev/ttyS3"
#define COM5 "/dev/ttyS4"
#define COM6 "/dev/ttyS5"

using namespace std;

/// @class ComSerie
/// @short Communication s�rie bloquante avec timeout.
class ComSerie {
public:
    /// @short  Constructeur.
    ComSerie();

    /// @short  Destructeur.
    ~ComSerie();

    /// @short  Permet d'ouvrir et de configurer un port serie.
    /// @param  port L'identificateur du port serie.
    /// @param  parametresCommunication Les param�tres associ�s (correspond
    ///         au champ c_cflag d'une structure termios (param�tres definis
    ///         dans "asm/termbits.h")
    /// @param  timeout Temps au d�l� duquel la reception est consid�r�e comme
    ///         un �chec. Exprim� en dixi�me de seconde.
    /// @return -1 si probl�me, 0 sinon.
    int ouvrir(string port, unsigned int parametresCommunication,
               unsigned int timeout);

    /// @short  Ferme le port s�rie et lui restaure ses param�tres initiaux.
    /// @return -1 si aucun port com n'�tait ouvert, 0 sinon.
    int fermer();

    /// @short Nettoie les donn�es pr�sentes sur le port com non lues.
    void flush();

    /// @short  Envoi une trame sur le port s�rie.
    /// @param  buf Un tableau d'octets contenant la trame � envoyer.
    /// @param  taille La taille du tableau en octet.
    /// @return
    int envoyer(const uint8_t *buf, const uint32_t taille);

    /// @short  Lit une trame sur le port s�rie.
    /// @param  buf Un tableau d'octets contenant la trame � recevoir.
    /// @param  tailleVoulue La taille en octet de la trame � recevoir.
    /// @return Si tous les octets attendus n'ont pas �t� recus avant un certain
    ///         temps d�fini par timeout_, retourne 0.
    ///         Sinon retourne le nombre d'octets effectivement lus.
    int recevoir(uint8_t *buf, const uint32_t tailleVoulue);


private:
    /// @short  Methode de configuration du port serie.
    void config_();

private:
    /// @short  Identificateur du port serie.
    string port_;

    /// @Nombre d'instance pour g�rer l'ouverture / fermeture du lien
    static int nb_instance_;

    /// @short  Parametres de communication correspondant au champ c_cflag d'une
    ///         structure termios (param�tres definis dans "asm/termbits.h")
    unsigned int parametresCommunication_;

    /// @short  Temps au d�l� duquel la reception est consid�r�e comme un �chec.
    unsigned int timeout_;

    /// @short  Structure de configuration du port serie.
    struct termios tios_;

    /// @short  Structure de configuration utilis�e pour sauvegarder
    ///         les anciens param�tres du port serie.
    static struct termios old_;

    /// @short  Descripteur pour le fichier associ� au port serie ouvert.
    int fd_;

    /// @short  Indicateur de l'�tat de la communication (ouverte ou non).
    bool comOuverte_;
};

#endif
