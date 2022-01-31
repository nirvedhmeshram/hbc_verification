# HBC verification steps
This repo provides steps and helper scripts needed to verify the correctness of the histogram binning calibration(HBC) implemented in torch-mlir. It has the following dependencies.
 - torch-mlir (https://github.com/llvm/torch-mlir)
 - FBGEMM/fbgemm_gpu/ (https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu)
 - IREE python bindings (https://google.github.io/iree/bindings/python/)
## Step 1 (Intermediate representation generation)
 - Go to the torch-mlir directory and check that it is built correctly
 ```shell
 cd ~/torch-mlir/
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
 - Note that if you wish to skip step 1 HBC_upstream.mlir is already provided in this repository that you may rename as HBC.mlir for the following steps
 ## Step 2 (Generating the ground truth input and output values from FBGEMM/fbgemm_gpu/)
 - Go to FBGEMM/fbgemm_gpu/directory and check that it is built correctly
 ```shell
 cd ~/FBGEMM/fbgemm_gpu/bench/
 python histogram_binning_calibration_benchmark.py
 ```
 - The output should look similar to this
 ```
INFO:root:hbc_cpu_torch.float16 time per iter: 277us
INFO:root:hbc_cpu_torch.float32 time per iter: 67us
INFO:root:hbc_cpu_torch.float64 time per iter: 87us
INFO:root:hbc_gpu_torch.float16 time per iter: 49us
INFO:root:hbc_gpu_torch.float32 time per iter: 36us
INFO:root:hbc_gpu_torch.float64 time per iter: 35us
INFO:root:hbc_by_feature_cpu_torch.float16 time per iter: 319us
INFO:root:hbc_by_feature_cpu_torch.float32 time per iter: 99us
INFO:root:hbc_by_feature_cpu_torch.float64 time per iter: 124us
INFO:root:hbc_by_feature_gpu_torch.float16 time per iter: 91us
INFO:root:hbc_by_feature_gpu_torch.float32 time per iter: 75us
INFO:root:hbc_by_feature_gpu_torch.float64 time per iter: 74us
INFO:root:generic_hbc_by_feature_cpu_torch.float16 time per iter: 715us
INFO:root:generic_hbc_by_feature_cpu_torch.float32 time per iter: 521us
INFO:root:generic_hbc_by_feature_cpu_torch.float64 time per iter: 520us
INFO:root:generic_hbc_by_feature_gpu_torch.float16 time per iter: 74us
INFO:root:generic_hbc_by_feature_gpu_torch.float32 time per iter: 70us
INFO:root:generic_hbc_by_feature_gpu_torch.float64 time per iter: 68us
```
- Go back to this repository and generate the input and output values so that they can be provided to the torch-mlir model
```shell
cd ~/hbc_verification
python histogram_binning_calibration_saveoutput.py
```
- Files hbc_inputs.pt and hbc_outputs.pt are generated that will be used to provide as input and check the output of the torch-mlir model in step 3
- Note that if you wish to skip step 2, hbc_inputs_upstream.pt and hbc_outputs_upstream.pt are already provided in this repository that you may rename for use in step 3

## Step 3 (Compiling the torch-mlir generated IR and running the compiled model)
- By running the histogram_binning_calibration_compileandrun.py you can verify the output of the two methods
```shell
histogram_binning_calibration_compileandrun.py
```
You should get the output shown below
```
success 1.1
success 1.2
```



 
 
 
