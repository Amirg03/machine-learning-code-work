# CHW4 · Principal component analysis

[Original assignment](assignment.pdf) · [Notebook](solution.ipynb)

Compare custom PCA with scikit-learn on Iris and inspect image reconstruction at different numbers of principal components. The notebook includes the saved reconstruction plots. To rerun that section, place `Superman.png` beside the notebook or update `IMAGE_PATH` to point to an image you can use.

## Assignment brief in English

Implement PCA and K-means without ready-made implementations. Compare PCA against a library result, reconstruct an RGB image with different component counts, and analyze clustering sensitivity to k and initial centers.

## How I solved it

The notebook compares custom PCA with scikit-learn on Iris, reconstructs each RGB channel and tracks reconstruction error and storage ratio. The custom K-means implementation clusters Iris using petal length and width and compares assignments, centers, and inertia across k and random seeds.
