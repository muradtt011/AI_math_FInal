

## Assignment Summary
This project investigates a central machine learning question: when does a one-hidden-layer nonlinear classifier genuinely improve upon a linear decision rule, and when is additional model complexity unnecessary?

The implementation compares softmax regression as a linear baseline with a one-hidden-layer neural network across three datasets with different geometric structures:

-Linear Gaussian dataset, which is approximately linearly separable,
-Moons dataset, which is inherently nonlinear,
-Digits dataset, which contains both linear and nonlinear structure.

The program includes:

=Core experiments comparing both models on all required datasets,
-Capacity ablation on the moons dataset using hidden widths {2, 8, 32},
-Optimizer comparison on the digits dataset using SGD, Momentum, and Adam,
-Failure-case analysis to study overfitting and class imbalance,
-Repeated-seed evaluation for robustness and stability,
-PCA/SVD-based input geometry analysis on the digits dataset,
-Implementation sanity checks such as gradient checking, probability validation, and tiny-subset training behavior.

This setup is designed to show where linear structure is sufficient, where nonlinearity provides meaningful benefits, and how model complexity, optimization, and data geometry jointly affect performance.

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

-If the program fails at startup, first check for syntax errors or filename mismatches. For example, ensure that softmax_classfication.py exists with the exact same spelling as used in the import statement.

-If a FileNotFoundError occurs, confirm that all .npz dataset files are present inside starter_pack/data/ and that the script is being run from the project root directory.

-If figures are not being saved, make sure the starter_pack/figures/ folder exists and that the program has permission to write into it.

-If plots do not display correctly, the selected matplotlib backend (TkAgg) may not be supported in your environment. In that case, switch to a compatible backend or run the script in a local Python environment with GUI support.

-If training appears unstable, check the learning rate and optimizer settings. Very large learning rates may lead to poor convergence or numerical instability.

-If repeated-seed or validation results look identical across runs, inspect how seeds are set and whether initialization is being reset as intended.

-If model performance seems unexpectedly weak, verify that data splits, normalization, and training parameters are consistent with the intended protocol.

For presentation and report preparation, use the saved figures from starter_pack/figures/ rather than screenshots taken manually, since saved figures are cleaner and more consistent.