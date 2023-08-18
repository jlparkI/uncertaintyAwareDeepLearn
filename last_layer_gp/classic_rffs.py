"""Implements vanilla random features using matrix multiplication. This is
sufficient if 1) the size of the latent representation is not large and
2) the number of random features desired is not large. Otherwise,
prefer fht_rffs.py."""
import math

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import Module

class VanillaRFFRegression(Module):
    """

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    Args:
        in_features: size of each input sample
        RFFs: The number of RFFs generated. Must be an even number. The larger out_features,
            the more accurate the approximation of the kernel, but also the greater
            the computational expense.
        out_targets: The number of output targets to predict. Generally this will
            be 1 if there is only one quantity you need the model to predict, but
            it can be > 1 if the model must predict multiple quantities. Defaults
            to 1 if not otherwise specified.
        random_seed: The random seed for generating the random features weight
            matrix. IMPORTANT -- always set this for reproducibility. Defaults to
            123.

    Shape:
        - Input: :math:`(N, H_{in})` where :math:`N` means number of datapoints.
          Only 2d input arrays are accepted.
        - Output: :math:`(N, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out_targets}`.

    Attributes:
        weight_mat: the non-learnable weights for generating random fourier features,
            of shape :math:`(H_{in}, 0.5 * RFFs)`
        output_weights: the learnable weights for generating the output predictions,
            of shape :math:`(RFFs, out_targets)`.

    Examples::

        >>> m = nn.VanillaRFFRegression(20)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 1])
    """
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_targets: int
    RFFs: int
    random_seed: int
    num_freqs: int
    feature_scale: float
    weight_mat: Tensor
    output_weights: Tensor

    def __init__(self, in_features: int, RFFs: int, out_targets: int=1,
            random_seed: int=123, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if not isinstance(out_targets, int) or not isinstance(RFFs, int) or \
                not isinstance(in_features, int):
            raise ValueError("out_targets, RFFs and in_features must be integers.")
        if out_targets < 1 or RFFs < 1 or in_features < 1:
            raise ValueError("out_targets, RFFs and in_features must be > 0.")
        if RFFs == 1 or RFFs % 2 != 0:
            raise ValueError("RFFs must be an even number.")

        self.in_features = in_features
        self.out_targets = out_targets
        self.RFFs = RFFs
        self.random_seed = random_seed
        self.num_freqs = int(0.5 * RFFs)
        self.feature_scale = math.sqrt(2. / float(self.num_freqs))

        self.register_buffer("weight_mat", torch.empty((in_features, self.num_freqs), **factory_kwargs))
        self.output_weights = Parameter(torch.empty((RFFs, out_targets), **factory_kwargs))
        self.reset_parameters()


    def reset_parameters(self) -> None:
        """Set parameters to initial values."""
        with torch.no_grad():
            rgen = torch.Generator()
            rgen.manual_seed(self.random_seed)
            self.weight_mat = torch.randn(rgen, self.weight_mat.size())
            self.output_weights[:] = 0.0


    def forward(self, input_tensor: Tensor) -> Tensor:
        """Forward pass. TODO: Should we add uncertainty calculation here,
        or in separate function?"""
        rff_mat = torch.zeros((input_tensor.shape[0], self.RFFs))
        intermediary = input_tensor @ self.weight_mat
        rff_mat[:,:self.num_freqs] = torch.sin(intermediary)
        rff_mat[:,self.num_freqs:] = torch.cos(intermediary)
        del intermediary
        return rff_mat @ self.output_weights

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_targets={self.out_targets}, RFFs={self.RFFs}, random_seed={self.random_seed}'
