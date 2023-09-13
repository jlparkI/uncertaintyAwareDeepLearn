"""Runs some simple basic functionality tests."""
import sys
import unittest
import torch

from uncertaintyAwareDeepLearn import VanillaRFFLayer


class TestVanillaRFFLayer(unittest.TestCase):
    """Tests the VanillaRFFLayer for basic functionality."""

    def test_error_checking(self):
        """Checks that inappropriate inputs cause the expected errors."""
        # Check first of all that an appropriate call to class constructor
        # does not raise any errors.
        try:
            my_layer = VanillaRFFLayer(in_features = 212, RFFs = 1024,
                    out_targets = 1, gp_cov_momentum = -1, gp_ridge_penalty = 1e-3,
                    likelihood = "gaussian", random_seed = 123)
        except:
            self.fail("Class constructor broken")

        with self.assertRaises(ValueError):
            my_layer = VanillaRFFLayer(in_features = 212, RFFs = 128,
                    out_targets = 2, gp_cov_momentum = -1, gp_ridge_penalty = 1e-3,
                    likelihood = "gaussian", random_seed = 123)
        with self.assertRaises(ValueError):
            my_layer = VanillaRFFLayer(in_features = 212, RFFs = 128,
                    out_targets = 1, gp_cov_momentum = -1, gp_ridge_penalty = 1e-3,
                    likelihood = "multiclass", random_seed = 123)
        with self.assertRaises(ValueError):
            my_layer = VanillaRFFLayer(in_features = 212, RFFs = 128,
                    out_targets = 1, gp_cov_momentum = -1, gp_ridge_penalty = 1e-3,
                    likelihood = "multiclass", random_seed = 123)
        with self.assertRaises(ValueError):
            my_layer = VanillaRFFLayer(in_features = 212, RFFs = 1,
                    out_targets = 1, gp_cov_momentum = -1, gp_ridge_penalty = 1e-3,
                    likelihood = "predict_stuff", random_seed = 123)
        with self.assertRaises(RuntimeError):
            my_layer = VanillaRFFLayer(in_features = 212, RFFs = 1024,
                    out_targets = 1, gp_cov_momentum = -1, gp_ridge_penalty = 1e-3,
                    likelihood = "gaussian", random_seed = 123)
            input_tx = torch.zeros((10,212))
            preds, var = my_layer(input_tx, get_var = True)


    def test_covariance_calc(self):
        """Checks that the covariance matrix is appropriately calculated
        when eval is called."""
        my_layer = VanillaRFFLayer(in_features = 212, RFFs = 256,
                    out_targets = 1, gp_cov_momentum = -1, gp_ridge_penalty = 1e-3,
                    likelihood = "gaussian", random_seed = 123)
        self.assertFalse(my_layer.fitted)
        covar_diag = torch.diag(my_layer.covariance)
        self.assertTrue(float(covar_diag.max()) == float(covar_diag.min()))
        self.assertTrue(float(covar_diag.max()) == 1 / my_layer.ridge_penalty)

        my_layer.eval()
        self.assertTrue(my_layer.fitted)
        self.assertTrue(torch.allclose(my_layer.covariance[0,0], torch.Tensor([999.999])))
        try:
            input_tx = torch.zeros((10,212))
            preds, var = my_layer(input_tx, get_var = True)
        except:
            self.fail("Class constructor broken")


if __name__ == "__main__":
    unittest.main()
