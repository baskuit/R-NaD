
# RL Experiments

Most people limited to consumer hardware are unable to test RL algorithms because of the cost of data-generation.

The repo is an implementation of DeepMind's R-NaD algorithm on a class of extremely vectorized imperfect information games, inspired by [Learning to Play Snake at One Million FPS](https://towardsdatascience.com/learning-to-play-snake-at-1-million-fps-4aae8d36d2f1).

Using this platform, anyone with a reasonably powered PC can experiment with the algorithm.

## Setup

    pip3 install -r requirements.txt
    python3 main.py

# R-NaD

Introduced here:

https://arxiv.org/abs/2002.08456

DeepNash (SOTA Stratego agent):

https://arxiv.org/abs/2206.15378

This new regularization allows neural network policies to converge to Nash equilibrium in imperfect information games, which previously were a well-known failure case for policy gradient.

# Trees

The imperfect information game implemented here is an abstract stochastic matrix tree. Many are familiar with the idea of a matrix game, like rock paper scissors. Imagine that, except sequential and with elements of chance.

The trees are randomly generated and can express wide range of depth and stochasticity using the provided numeric and functional parameters.

The default observation of a state is the matrix of expected payoff, under the assumption the rest of the game is played perfectly by both players.

This means our contrived game is well-behaved, in the sense that observations actually contain info about the optimal policy.


![Alt text](logs.png?raw=true "Title")