# HBC verification steps
This repo provides steps and helper scripts needed to verify the correctness of the histogram binning calibration(HBC) implemented in torch-mlir. It has the following dependencies.
 - torch-mlir (https://github.com/llvm/torch-mlir)
 - FBGEMM/fbgemm_gpu/ (https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu)
 - IREE python bindings (https://google.github.io/iree/bindings/python/)
## Step 1 (Intermediate representation generation)
 - Go to the torch-mlir directory and check that it is built correctly
 ```shell
 python -m e2e_testing.torchscript.main --filter HBC_basic
 ```
 The output should look like this
 ```
 PASS - "HBC_basic"

Summary:
    Passed: 1
```
 - If you are interested in viewing the HBC implementation open ~/torch-mlir/e2e_testing/torchscript/histogram_binning_calibration.py
 - Now clone this repository at the same level in the file system as torch-mlir and FBGEMM
 ```shell
 cd ..
 git clone https://github.com/nirvedhmeshram/hbc_verification
 ```
 - Go back to torch-mlir directory and generate the Intermediate Representation
 ``` shell
 cd torch-mlir
 python -m e2e_testing.torchscript.main --filter HBC_basic -c external --external-config "../hbc_verification/torchscript_e2e_config.py" -v
 ```
 - The intermediate representation HBC.mlir has been written in the directory ~/hbc_verification
 - Note that if you wish to skip step 1 HHC_upstream.mlir is already provided in this repository that you may rename as HBC.mlir for the following steps
 ## Step 2 (Generating the ground truth input and output values from FBGEMM/fbgemm_gpu/)
 - Go to FBGEMM/fbgemm_gpu/directory and check that it is built correctly
 ```shell
 cd 
 
 
 
 
 
