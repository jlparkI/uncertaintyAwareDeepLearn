# uncertaintyAwareDeepLearn

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
[the docs](https://github.com/jlparkI/uncertaintyAwareDeepLearn/docs/build/html/index.html)
