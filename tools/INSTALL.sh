## Downloads

https://www.qt.io/download-open-source#section-2
https://docs.conda.io/en/latest/miniconda.html
https://www.coppeliarobotics.com/downloads
http://wiki.ros.org/noetic/Installation/Ubuntu

mkdir Apps

## Softwares

sudo apt install -y dolphin
sudo apt install -y git
sudo apt install -y pluma
sudo apt install -y curl
sudo add-apt-repository ppa:inivation-ppa/inivation
sudo apt update
sudo apt install -y dv-gui
sudo apt install -y dv-runtime-dev
sudo snap install spotify

## Git

ssh-keygen -t ed25519

sudo git clone git@gitlab.ip.uca.fr:Alphat18/neuvisys-cpp.git
sudo git clone git@gitlab.ip.uca.fr:Alphat18/neuvisys-analysis.git
sudo git clone git@gitlab.ip.uca.fr:Alphat18/neuvisys-report.git

## Qt

sudo apt install -y libqt5charts5-dev

## Ros Noetic

sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
curl -sSL 'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC1CF6E31E6BADE8868B172B4F42ED6FBAB17C654' | sudo apt-key add -
sudo apt update
sudo apt install ros-noetic-desktop-full
source /opt/ros/noetic/setup.bash
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc

## Neuvisys

pip install empy
pip install catkin-pkg

cd neuvisys-cpp
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
cd ..

## Neuvisys-analysis

cd neuvisys-analysis
conda env create --file environment.yml
conda activate neuvisys-analysis
pip install pycryptodomex
spyder
