
# About

LHSGNN

# Installation

Requirements: Python (we recommend Python 3.8)

To install all required packages and libraries, you can use the install script. 

`./install.sh`


# Run SEAM

To train and test LHSGNN on predicting 3-clique on the USAir graph, use the following command:

`python Main.py --data-name USAir --prediction-method lhsgnn --motif clique --motif-k 3`
