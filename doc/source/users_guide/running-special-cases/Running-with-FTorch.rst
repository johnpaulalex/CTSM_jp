.. _running-with-ftorch-section:

#####################################
Running with Machine Learning Models via FTorch
#####################################

FTorch (Fortran interface to PyTorch) provides a lightweight bridge allowing CTSM Fortran routines to call PyTorch models at runtime for machine learning parameterizations, emulators, and data-driven subgrid processes.

Prerequisites
=============

1. **Python Environment**:
   Ensure you have created and activated the ``ctsm_pylib`` conda environment, which includes PyTorch (``pytorch-cpu``) for model training and TorchScript exporting:

   .. code-block:: bash

      ./py_env_create
      conda activate ctsm_pylib

2. **Checking out FTorch**:
   FTorch is an optional library submodule in CTSM. Check out FTorch using ``git-fleximod``:

   .. code-block:: bash

      git fleximod update FTorch

Workflow: Exporting PyTorch Models to TorchScript (.pt)
======================================================

FTorch requires models to be exported into TorchScript format (``.pt``). You can export your model using Python or the provided CTSM export tool:

.. code-block:: bash

   ./tools/ftorch/export_torchscript -o my_model.pt

Building CTSM with FTorch Support
=================================

To build CTSM with FTorch enabled, set ``USE_FTORCH=TRUE`` in your case configuration:

.. code-block:: bash

   ./xmlchange USE_FTORCH=TRUE
   ./case.setup
   ./case.build

Interfacing Fortran Code with FTorch
===================================

To load and execute a PyTorch model in Fortran code:

.. code-block:: fortran

   use ftorch, only : torch_model, torch_tensor, torch_model_load, &
                      torch_tensor_from_array, torch_model_forward, &
                      torch_kCPU, torch_delete

   type(torch_model) :: model
   type(torch_tensor) :: in_tensor(1), out_tensor(1)
   real(r8), target :: input_data(N), output_data(N)
   integer :: in_shape(1), out_shape(1)

   ! 1. Load TorchScript model file
   call torch_model_load(model, "my_model.pt", torch_kCPU)

   ! 2. Bind Fortran arrays to Torch tensors
   call torch_tensor_from_array(in_tensor(1), input_data, in_shape, torch_kCPU)
   call torch_tensor_from_array(out_tensor(1), output_data, out_shape, torch_kCPU)

   ! 3. Perform forward inference pass
   call torch_model_forward(model, in_tensor, out_tensor)

   ! 4. Clean up model and tensor handles
   call torch_delete(in_tensor(1))
   call torch_delete(out_tensor(1))
   call torch_delete(model)
