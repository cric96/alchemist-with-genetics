# AC + Neuro Evolution

This project provides an example of Neuroevolution and Aggregate Computing combination.

## Why

Aggregate computing aims at declaring global behaviour in a declarative way. So, why we should use something like Neuroevolution? If we use it we could lose expressiveness?

In Aggregate computing, despite its great power, happens to build building blocks that fail
in a real-case application (one of the most problematic blocks is Sparse Choice).
So, at the begging of this exploration, we think to use Learning (or in the general case an algorithm able to adapt) to improve some flaw blocks.

The expressiveness could be in danger because some building block became black blocks. But, the other
application logic, remain the same, so we don't lose it at all.

## What you can find in this repository

I use [Alchemist]() as a simulator to evaluate the fitness function of an aggregate program.
The target program is simple (the classic hop count).

As a Neuroevolution algorithm, I choose NEAT (Neuroevolution of augmenting topologies) because it is
simple and intuitive. Furthermore, in literature is used in different work with good results.

## Results

Currently, there isn't any good result found. The fitness goes to 0 but, when I try the network in 
a test set, the algorithm has not generalized well.