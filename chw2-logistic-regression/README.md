# CHW2 · Logistic regression

[Original assignment](assignment.pdf) · [Notebook](solution.ipynb) · [Model code](LogisticRegression.py)

Implement binary logistic regression and compare the results with scikit-learn. To rerun, place `Covid_pula_pula_bompa.csv` from the original course package beside the notebook. The full survey CSV is not bundled.

## Assignment brief in English

Build binary logistic regression and its loss from scratch with NumPy. Handle missing data, use a fixed 80/20 split, compare learning rates and a scikit-learn baseline, and revisit performance after correlation-based feature selection.

## How I solved it

The notebook fits preprocessing on the training set, trains the custom gradient descent model, and reports loss, ROC, accuracy, precision, and recall. It compares against scikit-learn on the same split and runs a second experiment with a correlation-filtered feature set.
