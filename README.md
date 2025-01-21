# uncertaintyAwareDeepLearn

*IMPORTANT NOTE* This project is deprecated as of 11/2024 and is being
merged with [resp_protein_toolkit](https://github.com/Wang-lab-UCSD/RESP2).
All of the functionality available in this repo is now available in
`resp_protein_toolkit` which can be easily installed using pip and contains
significant additional functionality not available in this project. We recommend
using `resp_protein_toolkit` in future.

A PyTorch class that implements an approximate Gaussian process as the last
layer of a neural network - compatible with any architecture and with regression,
binary logistic classification and classification. This provides a simple way to
obtain uncertainty calibration.

We recommend using this in combination with spectral normalization which is approximately
distance-preserving (see [Liu et al](https://arxiv.org/pdf/2205.00403.pdf) for
details). This ensures that datapoints far from the training set *in the input space*
are appropriately associated with high uncertainty. We may add standard
spectral-normalized layers to a future release to make this easier to implement.

For details on installation and usage, see
[the docs](https://jlparki.github.io/uncertaintyAwareDeepLearn/)
