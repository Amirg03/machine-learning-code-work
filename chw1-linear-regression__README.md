# CHW1 · Linear regression

[Original assignment](assignment.pdf) · [Notebook](solution.ipynb) · [Model code](LinearRegression.py)

Implement linear regression and use it to predict housing prices. The notebook covers feature scaling, fitting and evaluation with the included `housing.csv`.

## Assignment brief in English

Implement a linear regression class for California housing prices. Compute MSE gradients manually, compare several learning rates, and solve the same problem with the normal equation. Explain why a pseudoinverse can be preferable to an explicit inverse.

## How I solved it

The custom class implements both fitting methods. The notebook scales features, plots training loss for learning rates 0.001, 0.01, and 0.05, and compares learned weights and test RMSE for gradient descent and the closed-form solution. It also compares inverse and pseudoinverse estimates.
