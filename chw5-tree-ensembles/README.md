# CHW5 · Decision trees and ensembles

[Original assignment](assignment.pdf) · [Notebook](solution.ipynb)

Vary decision-tree depth on the Cryotherapy dataset, inspect feature importance, and compare Decision Tree, Random Forest, and AdaBoost using metrics, training times, and plots. The notebook reads the included `Cryotherapy.xlsx` via `openpyxl`.

## Assignment brief in English

Use an 80/20 split of Cryotherapy, evaluate tree depths 2, 4, 6, 8, and unrestricted, select a depth by F1, and compare the decision tree against Random Forest and AdaBoost. Report training time and the effect of estimator count.

## How I solved it

The notebook applies preprocessing, reports accuracy and F1, visualizes feature importance and the selected tree, and compares the three classifiers with training times. It plots ensemble accuracy for estimator counts from 5 to 200.
