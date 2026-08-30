#!/usr/bin/env python3
"""
Utility script to export PyTorch models to TorchScript (.pt) format for FTorch integration in CTSM.
"""

import sys
import argparse
import logging

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


if HAS_TORCH:
    class SampleIdentityModel(nn.Module):
        """
        Sample Identity Model used for testing and demonstration.
        Returns input unchanged.
        """
        def __init__(self):
            super().__init__()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * 1.0
else:
    class SampleIdentityModel:
        pass



def export_model_to_torchscript(model, output_file, method="script", example_input=None):
    """
    Export a PyTorch nn.Module to a TorchScript file.

    :param model: PyTorch nn.Module instance
    :param output_file: Path to output .pt file
    :param method: 'script' or 'trace'
    :param example_input: Tensor input required if method is 'trace'
    """
    if not HAS_TORCH:
        raise RuntimeError("PyTorch is not installed in the active environment. Please activate ctsm_pylib.")

    model.eval()
    if method == "script":
        scripted_model = torch.jit.script(model)
    elif method == "trace":
        if example_input is None:
            raise ValueError("example_input is required when method='trace'")
        scripted_model = torch.jit.trace(model, example_input)
    else:
        raise ValueError(f"Unknown export method: {method}. Must be 'script' or 'trace'.")

    scripted_model.save(output_file)
    logger.info("Successfully exported TorchScript model to: %s", output_file)
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Export PyTorch model to TorchScript .pt format for FTorch.")
    parser.add_argument("-o", "--output", default="simple_identity.pt", help="Output .pt file path (default: simple_identity.pt)")
    parser.add_argument("-m", "--method", choices=["script", "trace"], default="script", help="Export method (default: script)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    if not HAS_TORCH:
        print("ERROR: PyTorch is not available. Ensure PyTorch is installed in your python environment (ctsm_pylib).", file=sys.stderr)
        sys.exit(1)

    model = SampleIdentityModel()
    export_model_to_torchscript(model, args.output, method=args.method)
    print(f"Exported sample TorchScript model to: {args.output}")


if __name__ == "__main__":
    main()
