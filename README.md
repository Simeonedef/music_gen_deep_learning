<h1 align="center">
  Music Generation Using Deep Learning
</h1>
<p align="center">Sequence generation using LSTM <br>
This was developped as part of my bachelor thesis at The University of Manchester</p>
<div align="center"><a href="https://github.com/simeonedef"><img alt="@simeonedef" src="https://img.shields.io/badge/Author-Sim%C3%A9one%20de%20Fremond-lightgrey.svg" /></a>
<a href="https://opensource.org/licenses/MIT/"><img alt="License MIT" src="https://img.shields.io/badge/license-MIT-blue" /></a>
<a href="https://www.python.org/downloads/"><img alt="Python 3.6" src="https://img.shields.io/badge/python-3.6-green" /></a>
<img alt="Release date: june 2019" src="https://img.shields.io/badge/release%20date-june%202019-brightgreen" />
</div>

## Requirements
* Python 3.6
* Tensorflow==1.12.0
* Keras==2.2.4
* H5py==2.9.0
* Music21
* PyGame and [PyGame Menu](https://github.com/ppizarror/pygame-menu) (if you want to use generate_GUI.py)

More information in the requirements.txt file

## Setting up the project
To install all the required packaged simply run from the project's root directory:
````sh
pip install -r requirements.txt
````
See next steps to run the model and generate some music!

## Usage example
### Training the model
You need to first train the model which you can do by running one of the training scripts like so:
```sh
python train_single.py
```
if you to have only a single class as output, which means only one note playing at a time; otherwise:
```sh
python train_multiclass.py
```
if you want to take into account multiple notes being played at once.

Both training scripts will use every midi file in ./midi to train the network. Currently the directory contains all the pieces from Bach's Well-Tempered Clavier II as well as some additional files that you can decide to use if you enable their usage in the train_single.py file.

### Generating music
Once you have trained the network, you will have a set of weights which you can use instead of the weights.hdf5 file that comes in the repository. You can then generate using the trained network using generate.py like so:
```sh
python generate.py
```
The output will be test_output.mid. There is already an included midi file as an example of what the network can generate.

Additionally there is a GUI that you can use that will generate and play the music provided you have installed PyGame and PyGameMenu which you can run like so:
```bash
python generate_GUI.py
```

## Going forward
Below is a list of a few more things I would like to implement in this project:
* [x] finishing up train_output.py and getting a set of weights to generate music with multiple notes playing at once (in another repo)
* [ ] making multiple set of weights for different genres of music
* [x] implementing an option to generate fully random music
* [x] turning it into a web app to be accessible from everywhere [Click here !](https://deepmusic.herokuapp.com/)
