

## Assignment Summary
This project explores a fundamental question in machine learning: under what conditions does a one-hidden-layer nonlinear classifier meaningfully outperform a linear decision rule, and when is additional model complexity unnecessary?

The implementation compares softmax regression as a linear baseline with a one-hidden-layer neural network across three datasets with distinct geometric properties:

-A linear Gaussian dataset, which is approximately linearly separable,
-A moons dataset, which exhibits inherently nonlinear structure,
-A digits dataset, which contains a combination of linear and nonlinear patterns.

The project includes:

-Core experiments evaluating both models across all required datasets,
-Capacity ablation on the moons dataset using hidden layer widths {2, 8, 32},
-Optimizer comparison on the digits dataset using SGD, Momentum, and Adam,
-Failure-case analysis to examine overfitting and class imbalance,
-Repeated-seed evaluation to assess robustness and stability,
-PCA/SVD-based analysis to study input geometry in the digits dataset,
-Implementation sanity checks, including gradient verification, probability validation, and training behavior on small subsets.

This experimental framework is designed to identify when linear structure is sufficient, when nonlinear representations provide clear advantages, and how model capacity, optimization strategies, and data geometry together influence overall performance.

## Requirements
Python 3.x
The following Python libraries:
numpy
matplotlib
scikit-learn
scipy
The project directory should contain the following structure:
starter_pack/
├── data/
│   ├── digits_data.npz
│   ├── digits_split_indices.npz
│   ├── linear_gaussian.npz
│   └── moons.npz
│
├── figures/                 # Generated plots from experiments
│
├── results/                 #
│   
│
├── src/
│   ├── main.py
│   ├── nn.py
│   ├── softmax_classfication.py
│   ├── validation.py
│   ├── utils.py
│   └── pca.py
│
├── report/
│   ├── FinalReport.pdf
│          
│
├── slides/
│   └── FinalSlides.pptx
│
├── README.md

The figures/ folder should exist before saving output plots, or the program should be given permission to create and write files in that location.

## Execution Instructions

-Download or clone the project and unzip it if necessary.
-Make sure Python 3.x is installed on your machine.
-Install the required dependencies if they are not already available:

    pip install numpy matplotlib scikit-learn scipy

-Open a terminal (or command prompt) in the project root directory.
-Run the main script using the desired experimental flag. Since the project is executed from the root using the starter_pack/src/main.py path, the commands are:

    python starter_pack/src/main.py --rce

This runs the required core experiments.

    python starter_pack/src/main.py --ra

This runs the required ablations, including:
-capacity ablation on moons,
-optimizer study on digits,
-failure-case analysis.

    python starter_pack/src/main.py --isc

This runs the implementation sanity checks, including:

-gradient sanity check,
-tiny-subset loss decrease,
-tiny-subset overfitting behavior,
-probability-sum validation.

    python starter_pack/src/main.py --pca

This runs the PCA/SVD and input geometry analysis, including:

-scree plot,
-fixed PCA-dimension comparisons,
-2D PCA visualization for digits.

    python starter_pack/src/main.py --rss

This runs the repeated-seed statistics experiments.

Multiple experiment groups can also be executed together, for example:
    python starter_pack/src/main.py --cpad

The generated figures are saved automatically into the starter_pack/figures/ directory.

## Included Files

--main.py: Main execution file containing the experiment pipeline and plotting logic.

--nn.py: Implementation of the one-hidden-layer neural network, including training, prediction, backpropagation, optimization, and visualization methods.

-softmax_classfication.py: Implementation of softmax regression, including training, prediction, gradient checking, and loss visualization.

-validation.py: Validation and evaluation utilities, including best-epoch selection, repeated-seed evaluation, and reporting.

-utils.py: Helper functions such as accuracy computation, softmax, one-hot encoding, class imbalance generation, and histogram plotting.

-pca.py: PCA/SVD-related functions, including scree plotting and dimensionality reduction support.
digits_data.npz, digits_split_indices.npz, linear_gaussian.npz, moons.npz: Dataset files used throughout the project.

-figures/: Directory where generated plots and experiment outputs are saved.

README.md: Project description, setup instructions, and execution guide.

-FinalREPORT.pdf: Final written report containing mathematical background, methods, experiments, discussion, limitations, conclusion, and appendix.

-FinalSlides.pptx : Technical pitch summarizing the project question, methods, results, and insights.


## Configuration Options

You can modify the settings inside the source files, especially in main.py, nn.py, and softmax_classfication.py, to control the behavior of the experiments.

Useful configurable parameters include:

-epochs: Number of training epochs.
-batch_size: Mini-batch size used during training.
-learning_rate: Learning rate for optimization.
-lamda: L2 regularization strength.
-optimizer: Optimization method for the neural network (sgd, momentum, adam).
-size: Hidden layer width for the neural network, especially relevant for capacity ablation.
PCA dimensions in the fixed-dimension comparison, such as {10, 20, 40}.
Failure-case construction parameters in helper functions such as make_bias().

These options make it possible to test different learning conditions, observe changes in decision boundaries, and explore the relationship between optimization, representation, and generalization.

## Debugging & Tips

-If the program fails during startup, begin by checking for syntax issues or mismatched filenames. For instance, verify that softmax_classfication.py exists and matches the exact spelling used in the import.

-If you encounter a FileNotFoundError, ensure that all required .npz dataset files are located in starter_pack/data/ and that you are executing the script from the project’s root directory.

-If figures are not being saved, confirm that the starter_pack/figures/ directory exists and that the program has write permissions for that folder.

-If plots are not displaying properly, the chosen matplotlib backend (TkAgg) may not be supported in your environment. Consider switching to a compatible backend or running the script in a local setup with GUI support.

-If the training process seems unstable, review the learning rate and optimizer configuration. Excessively high learning rates can cause poor convergence or numerical issues.

-If repeated runs (e.g., with different seeds or validation) yield identical results, check how random seeds are set and whether model initialization is properly reset each time.

-If model performance is lower than expected, double-check data splits, normalization steps, and training parameters to ensure they align with the intended setup.

-For reports and presentations, always use the figures saved in starter_pack/figures/ instead of manually taken screenshots, as they provide better clarity and consistency.