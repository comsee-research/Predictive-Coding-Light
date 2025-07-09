# Predictive Coding Light network Analisys Library.

## Requirements

A python envrionment.
Alternatively, the library is made to be used with poetry.

### Installation with Poetry

- Install poetry:
``curl -sSL https://install.python-poetry.org | python3 -``

- Install the dependencies from the library folder:
``poetry install``

- You can then either launch a script with ``poetry run python your_script.py`` or activate the virtual environment with ``poetry shell``.

## Python Packages

Here is the list of packages and what they do:

- events: a package to create, modify and convert event files.
- spiking_network: a package to modify, visualize and launch the PCL network.

## Jupyter notebooks

You will find 6 notebooks: 
- Npz to Video: where you can load any npz event file and convert it to a gif or mp4 video.

- Orientation tuned and cross orientation suppressions: where you can use our saved data to reproduce the orientation-tuned and cross-orientation suppression effects observed in the PCL paper.

- RFs and Gabor fitting: where you can load a PCL network and see the learnt receptive fields as well as fitting them to gabor filters. Generated gabor fits and learnt receptive fields can be found in the "image" and "figure" folders of the network.

- Surround suppression: where you can use our saved data to reproduce the surround suppression suppression effects observed in the PCL paper.

- SVM_spikes_bins: where you can use our saved descriptors of binned spikes to train a SVM classifier and test it.

- tuningCurves: where you can use the saved responses of our simple and complex cells to generate the tuning curves observed in the PCL paper.