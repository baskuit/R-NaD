# R-NaD
Proof of concept illustration of the DeepNash alogrithm presented here:
https://arxiv.org/abs/2206.15378

game.py
Defines a class of stochastic matrix tree games with pre calculated Nash Equilibrium strategies.
Also defined is the Episodes class, which represents a batch of full length episodes.

metric.py
Defines function to measure the exploitablity ("NashConv") of a net-parameterized policy

rnad.py
Defines RNaD class which contains all the hyper parameters used in the algorithm, as well as saves and loads checkpoints in the project directory

![Alt text](logs.png?raw=true "Title")