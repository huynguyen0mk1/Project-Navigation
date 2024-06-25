# Project Navigation

## Project Details
The environment is provided by Udacity and is based on Unity ML-Agents. The agent's objective is to collect as many yellow bananas as possible while avoiding blue bananas.For this project, you will train an agent to navigate (and collect bananas!) in a large, square world.
A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of your agent is to collect as many yellow bananas as possible while avoiding blue bananas.The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:
0 - move forward.
1 - move backward.
2 - turn left.
3 - turn right.
The task is episodic, and in order to solve the environment, your agent must get an average score of +13 over 100 consecutive episodes.

## Getting Started
To set up your python environment to run the code in this repository, follow the instructions below.

1. **Install required dependencies.** You can install all required dependencies using the provided `setup.py` and `requirements.txt` file.

    ```bash
    !pip -q install .
    ```

2. **Download the Unity Environment.** Select the environment that matches your operating system from one of the links below:
    - [Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
    - [Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
    - [Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
    - [Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

3. **Unzip the environment file and place it in the root directory of this repository.**

## Instructions
To run the code, follow the Getting started instructions, git clone this repository and go to the folder repository. Then just type:

python Test.py