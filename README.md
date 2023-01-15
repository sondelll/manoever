# Manöver
_Autonomous driving with reinforcement learning_

## Prerequisites
* Python 3.9
* Tensorflow 2.10
* BeamNGPy
* BeamNG.tech 0.27.x
* Nvidia GPU (A 12 GB 30-series card was used during development)

## Getting Started
After cloning/downloading a copy of the repository
1. Install the BeamNG.tech software and take note of the installation path.
2. Make sure you've launched the program at least once and figure out where your "user folder" is, ("C:/Users/YOURNAME/AppData/Local/" is a good starting point, it shouldn't be hard to find, otherwise consult the beamNG documentation)
3. From the "misc_assets" folder in this repo, copy manoever.zip into the mods folder under your user folder (This contains the test tracks and the car configuration file)
4. In your shell, from the repository root folder, run `conda env create` to create an Anaconda environment (`conda activate manoever` to activate environment)
5. In mnvr.yaml, change the values to reflect your installation.
6. With the manoever environment active, run `python scripts/train.py` to try training a model.
