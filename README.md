### The Capability of this model was not too much, so needed a lot of improvement - see Vinci 2.0

# Vinci-Creator
A first step towards creativity in Robots

This project modifies Google's SketchRNN architecture and converts it into a Deep Reinforcement Learning(DRL) model which can be applied to robots. This modified architecture is based on the Deep Deterministic Policy Gradients(DDPG) setup, with a SketchRNN-Decoder based actor and critic. The agents have been modified to work with a custom gym environment that motivates a particular direction of sketches based on the reward function.

## Literature:

[A Neural Representation of Sketch Drawings](https://arxiv.org/pdf/1704.03477.pdf) - SketchRNN

![image](https://user-images.githubusercontent.com/43128490/119958008-3c3a1e80-bfc0-11eb-9a45-7b9868a227a0.png)

[CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING](https://arxiv.org/pdf/1509.02971.pdf) - DDPG

![image](https://user-images.githubusercontent.com/43128490/119958112-58d65680-bfc0-11eb-96c0-07868ddd7e76.png)

## Code References:

[Keras Implementation of SketchRNN](https://github.com/MarioBonse/Sketch-rnn)

[Google's SketchRNN Implementation](https://github.com/magenta/magenta/tree/master/magenta/models/sketch_rnn)

[Keras.io Tutorial on DDPG](https://keras.io/examples/rl/ddpg_pendulum/)

[Create Custom Gym Environments](https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e)

## Repository Information:

-> There are two saved models which are pretrained on two different reward function: 

1. Pixel Maximizer - High reward for a bigger image
2. Black Pixel Maximizer - High reward for a higher black to white pixel ratio.

-> Relevant files:

1. `sketch_rnn.py` - This module defines the encoder and decoder models based on the SketchRNN architecture. This can be used to create the actor, critic and the environment.
2. `HyperParameters.py` - This is a class which is used to define all the hyper parameters used to train and/or test the model. This class defines hyper parameters relevant to both DDPG and SketchRNN, as well as the general global variables such as number of training epochs, learning rates, etc.
3. `draw.py` - This is a utility module, with helper functions used to draw generated strokes in SVG format and save them to PNG formats.
4. `ddpg1.ipynb` - This is the Jupyter Notebook which carries out the main training and generating of the model. It defines the environment, replay buffer, action noise, etc. that are required in the training and testing.

-> Folders:

1. `model` - contains the pretrained models for the two rewards as well as the pretrained model weights of SketchRNN from the repository 
2. `results` - contains the results of training the model such as a plot of reward vs episodes, intermediate drawings rendered during training, etc.
3. `data` - This just contains a single reference image used by the environment to give an initial condition to the model. This does NOT contain training data as the training data is generated during the training as a part of DDPG functionality.

## System Requirements:

1. `Python 3.8`
2. `Tensorflow 2.5.0`
3. `svglib 1.0.1`
4. `svgwrite 1.4.1`
5. `tensorflow-probability 0.12.2`
6. `numpy 1.19.5`
7. `gym 0.18.0`
8. `openCV-python 4.5.1.48`
9. `matplotlib 3.3.4`
10. `jupyter-notebook 6.3.0`

## Using the Repository:

### Running Locally - (Not Recommended unless you have a really strong GPU setup)

1. Clone the repository on to your local device.
2. Change the Hyperparameters in the respective python file as desired.
3. Open jupyter notebook in the repository directory. Open the notebook `ddpg1.ipynb`.
4. a) To use a pretrained model - set the `Model_Name` parameter to "Pixel Maximizer" or "Black Pixel Maximizer" and the flag `continue_training = True`.

   b) To train a new model - set the `Model_Name` to the desired name, the flag `continue_training = False` and edit the `_get_reward()` function in the SketchEnvironment according to your requirements.
   
   c) You can also retrain the model for the default two reward functions by setting the appropriate `Model_Name` and `continue_training = False`.
5. Run the notebook cell by cell to perform training, followed by plotting and generating. For local running, you can skip the drive mounting steps and instead of the default working directory, use your desired working directory.
6. The `train()` function can be used to run the training loop for the number of episodes desired (number of episodes is a hyper parameter set in the `HyperParameters.py`). This functions returns a list of episodic reward, which is used by the `plot_reward()` function to plot and save the reward.
7. the `get_ground_truth()` function can be used to generate a sketch. This function uses the target actor and runs a single episode without any action noise. This function generates a numpy array for "strokes" which can be fed as inputs to the `draw_strokes()` function in the module `draw.py` to get the final output sketch.
8. the `get_setup()` function sets up the replay buffer, sketch environment, actor, critic, target_actor, target_critic, encoderRNN and action noise that are all required for the training loop.

### Running on Google Colab - (Recommended, with GPU runtime)

1. Add the repository to drive (all sub folders and files intact), open `ddpg1.ipynb` on google colab and follow the steps above.
2. Here you should mount your drive (to make use of the python modules). Change the `working_directory` to `/content/drive/MyDrive` instead of the default `/content/drive/Shareddrives/Vinci`. 
3. Everything else is just the same as running locally.

## Implementation Details:

Check out the PDF titled 'Report.pdf' in the repository.
