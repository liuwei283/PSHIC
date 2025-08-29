# Hi-C Phase Analysis Tool

This repository contains a tool for performing Hi-C phase analysis, which is a computational method used to analyze chromatin interactions within the genome. The tool is designed to process Hi-C data, identify outliers, normalize the data, and optimize phase matrices using a power-law model.

## Features

- **Outlier Detection**: Identifies and removes outlier bins from the Hi-C matrix.
- **Normalization**: Performs Knight-Ruiz (KR) normalization on the Hi-C matrix.
- **Optimization**: Utilizes a power-law model to optimize phase matrices.
- **Command-line Interface**: Easily configurable through command-line arguments.

## Installation

To set up the environment for this tool, use the provided `environment.yml` file with Conda:
