# Training highly effective connectivities within neural networks with randomly initialized, fixed weights


This repository contains the code for the experiments from this ["paper"](https://arxiv.org/abs/2006.16627)

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



