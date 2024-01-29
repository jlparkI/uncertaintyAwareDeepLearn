"""Implements vanilla random features using matrix multiplication. This is
sufficient if the size of the latent representation is not large."""
import math

import torch
from torch import Tensor
from torch.nn.parameter import Parameter
from torch.nn import Module


_ACCEPTED_LIKELIHOODS = ("gaussian", "binary_logistic", "multiclass")


class VanillaRFFLayer(Module):
    """
    A PyTorch layer for random features-based regression, binary classification and
    multiclass classification.

    Args:
        in_features: The dimensionality of each input datapoint. Each input
            tensor should be a 2d tensor of size (N, in_features).
        RFFs: The number of RFFs generated. Must be an even number. The larger RFFs,
            the more accurate the approximation of the kernel, but also the greater
            the computational expense. We suggest 1024 as a reasonable value.
        out_targets: The number of output targets to predict. For regression and
            binary classification, this must be 1. For multiclass classification,
            this should be the number of possible categories in the data.
        gp_cov_momentum (float): A "discount factor" used to update a moving average
            for the updates to the covariance matrix. 0.999 is a reasonable default
            if the number of steps per epoch is large, otherwise you may want to
            experiment with smaller values. If you set this to < 0 (e.g. to -1),
            the precision matrix will be generated in a single epoch without
            any momentum.
        gp_ridge_penalty (float): The initial diagonal value for computing the
            covariance matrix; useful for numerical stability so should not be
            set to zero. 1e-3 is a reasonable default although in some cases
            experimenting with different settings may improve performance.
        likelihood (str): One of "gaussian", "binary_logistic", "multiclass".
            Determines how the precision matrix is calculated. Use "gaussian"
            for regression.
        amplitude (float): The kernel amplitude. This is the inverse of
            the lengthscale. Performance is not generally
            very sensitive to the selected value for this hyperparameter,
            although it may affect calibration. Defaults to 1.
        random_seed: The random seed for generating the random features weight
            matrix. IMPORTANT -- always set this for reproducibility. Defaults to
            123.

    Shape:
        - Input: :math:`(N, H_{in})` where :math:`N` means number of datapoints.
          Only 2d input arrays are accepted.
        - Output: :math:`(N, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out}` = out_targets.

    Examples::

        >>> m = nn.VanillaRFFLayer(20, 1)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 1])
    """

    def __init__(self, in_features: int, RFFs: int, out_targets: int=1,
            gp_cov_momentum = 0.999, gp_ridge_penalty = 1e-3,
            likelihood = "gaussian", amplitude = 1.,
            random_seed: int=123, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        if not isinstance(out_targets, int) or not isinstance(RFFs, int) or \
                not isinstance(in_features, int):
            raise ValueError("out_targets, RFFs and in_features must be integers.")
        if out_targets < 1 or RFFs < 1 or in_features < 1:
            raise ValueError("out_targets, RFFs and in_features must be > 0.")
        if RFFs <= 1 or RFFs % 2 != 0:
            raise ValueError("RFFs must be an even number greater than 1.")
        if likelihood not in _ACCEPTED_LIKELIHOODS:
            raise ValueError(f"Likelihood must be one of {_ACCEPTED_LIKELIHOODS}.")
        if likelihood in ["gaussian", "binary_logistic"] and out_targets != 1:
            raise ValueError("For regression and binary_logistic likelihoods, "
                    "only one out target is expected.")
        if likelihood == "multiclass" and out_targets <= 1:
            raise ValueError("For multiclass likelihood, more than one out target "
                    "is expected.")

        self.in_features = in_features
        self.out_targets = out_targets
        self.fitted = False
        self.momentum = gp_cov_momentum
        self.ridge_penalty = gp_ridge_penalty
        self.RFFs = RFFs
        self.likelihood = likelihood
        self.amplitude = amplitude
        self.random_seed = random_seed
        self.num_freqs = int(0.5 * RFFs)
        self.feature_scale = math.sqrt(2. / float(self.num_freqs))

        self.register_buffer("weight_mat", torch.zeros((in_features, self.num_freqs), **factory_kwargs))
        self.output_weights = Parameter(torch.empty((RFFs, out_targets), **factory_kwargs))
        self.register_buffer("covariance", torch.zeros((RFFs, RFFs), **factory_kwargs))
        self.register_buffer("precision", torch.zeros((RFFs, RFFs), **factory_kwargs))
        self.reset_parameters()


    def train(self, mode=True) -> None:
        """Sets the layer to train or eval mode when called
        by the parent model. NOTE: Setting the model to
        eval if it was previously in train will cause
        the covariance matrix to be calculated. This can
        (if the number of RFFs is large) be an expensive calculation,
        so expect model.eval() to take a moment in such cases."""
        if mode:
            self.fitted = False
        else:
            if not self.fitted:
                self.covariance[...] = torch.linalg.pinv(self.ridge_penalty *
                    torch.eye(self.precision.size()[0], device = self.precision.device) +
                    self.precision)
            self.fitted = True


    def reset_parameters(self) -> None:
        """Set parameters to initial values. We don't need to use kaiming
        normal -- in fact, that would set the variance on our sqexp kernel
        to something other than 1 (which is ok, but might be unexpected for
        the user)."""
        self.fitted = False
        with torch.no_grad():
            rgen = torch.Generator()
            rgen.manual_seed(self.random_seed)
            self.weight_mat = torch.randn(generator = rgen,
                    size = self.weight_mat.size())
            self.output_weights[:] = torch.randn(generator = rgen,
                    size = self.output_weights.size())
            self.covariance[:] = (1 / self.ridge_penalty) * torch.eye(self.RFFs)
            self.precision[:] = 0.


    def reset_covariance(self) -> None:
        """Resets the covariance to the initial values. Useful if
        planning to generate the precision & covariance matrices
        on the final epoch."""
        self.fitted = False
        with torch.no_grad():
            self.precision[:] = 0.
            self.covariance[:] = (1 / self.ridge_penalty) * torch.eye(self.RFFs)

    def forward(self, input_tensor: Tensor, update_precision: bool = False,
            get_var: bool = False) -> Tensor:
        """Forward pass. Only updates the precision matrix if update_precision is
        set to True.

        Args:
            input_tensor (Tensor): The input x values. Must be a 2d tensor.
            update_precision (bool): If True, update the precision matrix. Only
                do this during training.
            get_var (bool): If True, obtain the variance on the predictions. Only
                do this when generating model predictions (not necessary during
                training).

        Returns:
            logits (Tensor): The output predictions, of size (input_tensor.shape[0],
                    out_targets)
            var (Tensor): Only returned if get_var is True. Indicates variance on
                predictions.

        Raises:
            RuntimeError: A RuntimeError is raised if get_var is set to True
                but model.eval() has never been called."""
        if len(input_tensor.size()) != 2:
            raise ValueError("Only 2d input tensors are accepted by "
                    "VanillaRFFLayer.")
        rff_mat = self.amplitude * input_tensor @ self.weight_mat
        rff_mat = self.feature_scale * torch.cat([torch.cos(rff_mat), torch.sin(rff_mat)], dim=1)
        logits = rff_mat @ self.output_weights

        if update_precision:
            self.fitted = False
            self._update_precision(rff_mat, logits)

        if get_var:
            if not self.fitted:
                raise RuntimeError("Must call model.eval() to generate "
                        "the covariance matrix before requesting a "
                        "variance calculation.")
            with torch.no_grad():
                var = self.ridge_penalty * (self.covariance @ rff_mat.T).T
                var = torch.sum(rff_mat * var, dim=1)
            return logits, var
        return logits


    def _update_precision(self, rff_mat: Tensor, logits: Tensor) -> Tensor:
        """Updates the precision matrix. If momentum is < 0, the precision
        matrix is updated using a sum over all minibatches in the epoch;
        this calculation therefore needs to be run only once, on the
        last epoch. If momentum is > 0, the precision matrix is updated
        using the momentum term selected by the user. Note that for multi-class
        classification, we actually compute an upper bound; see Liu et al. 2022.;
        since computing the full Hessian would be too expensive if there is
        a large number of classes."""
        with torch.no_grad():
            if self.likelihood == 'binary_logistic':
                prob = torch.sigmoid(logits)
                prob_multiplier = prob * (1. - prob)
            elif self.likelihood == 'multiclass':
                prob = torch.max(torch.softmax(logits), dim=1)
                prob_multiplier = prob * (1. - prob)
            else:
                prob_multiplier = 1.

            gp_feature_adjusted = torch.sqrt(prob_multiplier) * rff_mat
            precision_matrix_minibatch = gp_feature_adjusted.T @ gp_feature_adjusted
            if self.momentum < 0:
                self.precision += precision_matrix_minibatch
            else:
                self.precision[...] = (
                    self.momentum * self.precision
                    + (1 - self.momentum) * precision_matrix_minibatch)
