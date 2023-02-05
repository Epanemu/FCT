# XCT_nn
#### eXplained Clasification Tree with NNs in leaves
An ML model with clasification tree optimized for leaf accuracy. This accuracy is then further improved by Neural Networks. (Work in Progress...)

The model combines the explainability of Classification Trees with the accuracy of Neural Nets.

## Examples

The scripts `highest_acc_tester*` perform a halving algorithm that seeks the highest accuracy,
for which a model can be found under an hour by the solver.

The script `gradual_depth_increase.py` performs a series of MIP optimization, where it increases the depth of the model,
while using the solution of previous tree as a warm start
