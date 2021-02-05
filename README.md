# Action Permutations
Predicting action sequences with visual permutations (This messy repository compliments the paper: *Burke, Subr and Ramamoorthy., Action sequencing using visual permutations, accepted at IEEE RA-L, 2021*. See [here](https://arxiv.org/abs/2008.01156) for pre-print)

Requirements:
- [PyRep](https://github.com/stepjam/PyRep) + [CoppeliaSim](https://www.coppeliarobotics.com/)
- PyTorch

Code to reproduce the results are in the three experiment folders:

- [Soma puzzle](./soma/)
- [Tower building](./tower/)
- [Scrabble](./scrabble/)

Models compared here include fully connected neural networks, temporal convolution neural networks (based on this [repository](https://github.com/locuslab/TCN)) and an action sequencing model using latent permutations (building on the work [here](https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch)).
