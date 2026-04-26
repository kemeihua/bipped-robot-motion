import torch

# load the trained policy jit model
policy_jit_path = f'/home/liangzhiyuan/RL/X02/x02-sim2real_new_grpc/scripts/policies/test/2025-04-02_16-56-02_x2_real_2000.pt'
policy_jit_model = torch.jit.load(policy_jit_path)

#set the model to evalution mode
policy_jit_model.eval()

# creat a fake input to the model
test_input_tensor = torch.randn(1, 40*15)

#specify the path and name of the output onnx model
policy_onnx_model = f'/home/liangzhiyuan/RL/X02/x02-sim2real_new_grpc/scripts/policies/test/2025-04-02_16-56-02_x2_real_2000.onnx'

#export the onnx model
torch.onnx.export(policy_jit_model,
                  test_input_tensor,
                  policy_onnx_model,   # params below can be ignored
                  export_params=True,
                  opset_version=11,
                  do_constant_folding=True,
                  input_names=['input'],
                  output_names=['action'],
                  )
# torch.onnx.export(policy_jit_model,
#                   test_input_tensor,
#                   policy_onnx_model,   # params below can be ignored
#                   export_params=True,
#                   opset_version=11,
#                   do_constant_folding=True,
#                   input_names=['input'],
#                   output_names=['action','est'],
#                   dynamic_axes={
#                       "input": {0: "batch_size"},  # 动态批次维度
#                       "action": {0: "batch_size"},
#                       "est": {0: "batch_size"}
#                   }
#                   )