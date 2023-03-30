# Measuring the VC-Dimension Using Experimental Method

This repository contains a Python implementation of "Measuring the VC-Dimension Using Optimized Experimental Design" by
Shao, Cherkassky, and Li (2000). This code measures the effective VC-dimension of a linear regression model using
experimental method.

## Installation

To run this code, you need to have Python 3.x installed on your system. You also need to have the following Python
packages installed:

- numpy
- scikit-learn

You can install these packages using the following command:

```sh
pip install numpy scikit-learn
```

## Usage

To use this code, you can simply run the `vc_dimension.py` file. This will generate a random dataset, train a linear
regression model on it, and estimate the effective VC-dimension of the model using experimental method.

```sh
python vc_dimension.py
```

The output will be the estimated effective VC-dimension of the linear regression model.

## References

- Shao, X., Cherkassky, V., & Li, W. (2000). Measuring the VC-dimension using optimized experimental Design. Neural
  Computation, 12(8), 1969-1986.