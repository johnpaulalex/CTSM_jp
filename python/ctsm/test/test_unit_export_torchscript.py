"""
Unit test for ctsm.machine_learning.export_torchscript
"""

import unittest
from ctsm.machine_learning.export_torchscript import SampleIdentityModel, export_model_to_torchscript, HAS_TORCH


class TestExportTorchscript(unittest.TestCase):
    def test_sample_identity_model(self):
        """Test instantiation of SampleIdentityModel."""
        model = SampleIdentityModel()
        self.assertIsNotNone(model)

    def test_has_torch_flag(self):
        """Test HAS_TORCH boolean flag."""
        self.assertIsInstance(HAS_TORCH, bool)


if __name__ == "__main__":
    unittest.main()
