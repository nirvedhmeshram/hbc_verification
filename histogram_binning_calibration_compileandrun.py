from iree.compiler import compile_str
from iree import runtime as ireert
import torch
inputfilepath="HBC.mlir"
with open(inputfilepath, "rb") as input_file:
    compiled_data = input_file.read()
flatbuffer_blob = compile_str(compiled_data, target_backends=["dylib-llvm-aot"])
vm_module = ireert.VmModule.from_flatbuffer(flatbuffer_blob)
backend = "dylib-llvm-aot"    
backend_config = "dylib"  
config = ireert.Config(backend_config)
ctx = ireert.SystemContext(config=config)
ctx.add_vm_module(vm_module)
HbcCompiled = ctx.modules.module
input_data=torch.load("hbc_inputs.pt")
[output1,output2]=HbcCompiled.forward(input_data['segment_values'].int().numpy(),input_data['segment_offsets'].int().numpy(),input_data['logits'].numpy())
input_data=torch.load("hbc_outputs.pt")
calibrated_prediction_golden1=input_data['feature_output1'];
torch.testing.assert_allclose(output1,calibrated_prediction_golden1,rtol=1e-03, atol=1e-03,)
print("success 1.1")
calibrated_prediction_golden1=input_data['feature_output2'];
torch.testing.assert_allclose(output2,calibrated_prediction_golden1,rtol=1e-03, atol=1e-03,)
print("success 1.2")


