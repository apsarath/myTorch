# myTorch

A pyTorch research framework for Memory Augmented Neural Networks and Reinforcement Learning.

### Major directories

* myTorch/utils - contains basic experiment manager, tensorboard logger, and util functions.
* myTorch/memory - contains memory cells like LSTM, GRU.
* myTorch/rllib - contains standard implementations of RL agents.
* myTorch/environment - contains several environments that can be linked to the RL agents.
* myTorch/task - contains data iterators for tasks used in the projects.
* myTorch/projects - contains projects based on myTorch.


### Setup

* run `pip install -r requirements.txt`
* set an environment variable called 'MYTORCH_SAVEDIR' to a folder where you want the output logs and model checkpoints to be saved.
* set an environment variable called 'MYTORCH_DATA' to a folder where you want to keep all your data.
