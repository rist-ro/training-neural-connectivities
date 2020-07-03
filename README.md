# Training highly effective connectivities within neural networks with randomly initialized, fixed weights


This repository contains the code for the experiments from the following publication:

C. Ivan & R. V. Florian (2020), [Training highly effective connectivities within neural networks with randomly initialized, fixed weights](https://arxiv.org/abs/2006.16627). arXiv:2006.16627.

### Abstract
We present some novel, straightforward methods for training the connection graph of a randomly initialized neural network without training the weights. These methods do not use hyperparameters defining cutoff thresholds and therefore remove the need for iteratively searching optimal values of such hyperparameters. We can achieve similar or higher performances than in the case of training all weights, with a similar computational cost as for standard training techniques. Besides switching connections on and off, we introduce a novel way of training a network by flipping the signs of the weights. If we try to minimize the number of changed connections, by changing less than 10% of the total it is already possible to reach more than 90% of the accuracy achieved by standard training. We obtain good results even with weights of constant magnitude or even when weights are drawn from highly asymmetric distributions. These results shed light on the over-parameterization of neural networks and on how they may be reduced to their effective size.

### Requirements
- [numpy](https://numpy.org/)
- [tensorflow](https://www.tensorflow.org/) (1.14.0)
- [matplotlib](https://matplotlib.org/)


### Run experiments

The default parameters run LeNet on MNIST with the free pruning method:
```markdown
python MaskTrainer.py
```

Plot results for the default experiment:
```markdown
python Plotter.py
```

An example of how to set parameters, for Conv6, minimal pruning, Signed He Constant distribution, relu activation, masking function, batch size, maximum training epochs:

```markdown
python MaskTrainer.py --nettype Conv6 --traintype MinPruning --initializer heconstant --activation relu --masktype mask --batchsize 25 --maxepochs 100 --seed 1234 --p1 0.5 --lr 0.003 --outputpath Outputs 
```



