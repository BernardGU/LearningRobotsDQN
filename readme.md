# Learning Robots DQN Project
## About The Project
This is the code used in the project Human Demonstrations to aid initial exploration of **Deep Q-Learning Model in sparse-reward Atari games** for the course **Learning Robots**.

It includes a learning agent that incorporates a double DQN closely following the architecture and sticking to the hyperparameters proposed in [Mnih, et. al. (2015)](https://www.nature.com/articles/nature14236)

It includes compatibility with the Atari game environments included in the [OpenAI Gym](https://github.com/openai/gym/). **WARNING:** gym[atari] currently supports only Linux OS, so this project was also originally built for Linux systems only.

It also includes a trainer class to perform the following experiments:
- Random play in an environment
- Playing with a specified epsilon-greedy policy
- Evaluate the agent in a given environment
- Generate samples from a human player
- Train the agent with human-generated samples (offline learning)
- Train the agent with Deep Q-Learning and experience replay

### Built With

* [OpenAI Gym](https://github.com/openai/gym/)
* [PyTorch](https://github.com/pytorch/pytorch)

<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* Linux OS or WSL
* conda
  ```sh
  $ conda --version
  conda 4.9.2
  ```
* **Note:** The replay memory buffer in this project implements by default a lazy loading feature based on threads to avoid holding all the transitions at once. Therefore, it does not require a high performance computer with lots of RAM to perform the experiments (although, it is still recommended for speed reasons).

### Installation

1. Clone the repo and cd to it
    ```sh
    git clone https://github.com/BernardGU/learning-robots-dqn-project.git
    cd learning-robots-dqn-project
    ```
2. Create a new conda environment with the required packages
    ```sh
    conda env create --name ENV_NAME --file env.yml
    ```
    or install the required packages in your prefered environment
    ```sh
    conda env update -n YOUR_EXISTING_ENV_NAME --file env.yml
    ```

3. Activate your environment
    ```sh
    conda activate ENV_NAME
    ```


## Usage

```sh
python main.py -g <GAME> -s <SESSION_NAME> -a <ACTION> -r <REPLAY_DIR> -c <CHECKPOINT_DIR>

    - GAME: name of the game
    - SESSION_NAME: name of the session (concatenates path: "./data/<GAME>/sessions/<SESSION_NAME>")
    - ACTION: 'play' | 'collect' | 'train_demo' | 'train_xp' | 'eval'
    - REPLAY_DIR: dir where the replay is saved (concatenates path: "./data/<GAME>/transitions/<REPLAY_DIR>")
    - CHECKPOINT_DIR: dir where the checkpoint was saved (add full path)
```

Some arguments are optional, depending on the use case. Following are example use cases:
### To train a new agent using Deep Q-Learning with experience replay
---
```sh
python main.py -g <GAME> -s <SESSION_NAME> -a 'train_xp' -r <REPLAY_DIR>
```
* Learning agent checkpoints will be stored in `./data/<GAME>/sessions/<SESSION_NAME>`
* Transitions will be saved in `./data/<GAME>/transitions/<REPLAY_DIR>`
* **Note:** One can also resume the training of an agent by using the same `<SESSION_NAME>` parameter
* **Note:** One can use any compatible (same game, same capacity) REPLAY_DIR
### To train a new agent using demonstrations from DEMO_DIR
---
```sh
python main.py -g <GAME> -s <SESSION_NAME> -a 'train_demo' -r <DEMO_DIR>
```
* Learning agent checkpoints will be stored in `./data/<GAME>/sessions/<SESSION_NAME>`
* Demonstrations will be loaded from `./data/<GAME>/transitions/<DEMO_DIR>`
* **Note:** One can also resume the training of an agent by using the same `<SESSION_NAME>` parameter

### To collect human demonstrations into DEMO_DIR
---
```sh
python main.py -g <GAME> -s <SESSION_NAME> -a 'collect' -r <DEMO_DIR>
```
* Demonstrations will be sotred in `./data/<GAME>/transitions/<DEMO_DIR>`
* **Note:** One can also resume an initiated collection of demonstrations by using the same `<DEMO_DIR>` parameter
* [Available Gravitar samples](https://drive.google.com/file/d/1seoeLgr1rvpgsx7zrEVzM6v90Wzcd_E5/view?usp=sharing)

### To play the game freely (for humans)
---
```sh
python main.py -g <GAME> -s <SESSION_NAME> -a 'play'
```

### To evaluate a learning agent
---
```sh
python main.py -g <GAME> -s <SESSION_NAME> -a 'eval' -c <CHECKPOINT_DIR>
```
* You need to provide the full path of a checkpoint directory
* The `eval_results.csv` file is generated in `./data/<GAME>/sessions/<SESSION_NAME>` folder

