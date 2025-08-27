# This example showcases how to leverage the 3.1 8B Instruct models using 
# torch.compile to accelerate inference.
#
# You need CUDA and torch >= 2.3 in order to run this example.

import os
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tt_torch.tools.utils import CompilerConfig
from tt_torch.tools.device_manager import DeviceManager
from tt_torch.dynamo.backend import backend, BackendOptions
os.environ["TOKENIZERS_PARALLELISM"] = "false" # silence warnings when compiling

device = DeviceManager.create_parent_mesh_device(mesh_shape=[1, 1])
ckpt = "meta-llama/Meta-Llama-3.1-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16, use_cache= True)
# float16 지원 안 됨. bfloat16 사용해야 함
# use_cache는 뭔가 어디에 저장되는 캐시인가 그리고 저장 대상은?

cc = CompilerConfig()
cc.enable_consteval = False
cc.consteval_parameters = False
options = BackendOptions()
options.compiler_config = cc
options.devices = [device]
buffer_cache = {}
options.buffer_cache = buffer_cache
constant_cache = {}
options.constant_cache = constant_cache

# model.to(device) # 이걸 안 쓰는 코드는 뭔가
#tt_mlir.device를 인식하지 않아서 에러가 난다.

tokenizer = AutoTokenizer.from_pretrained(ckpt, torch_dtype=torch.bfloat16) # torch_dtype을 명시하지 않아도 괜찮은가?
#tokenizer.pad_token을 지정하지 않아도 괜찮은가

prompt = "Why dogs are so cute?"
# inputs = tokenizer(prompt, return_tensors="pt").to(device)
inputs = tokenizer(prompt, return_tensors="pt")
# tokenizer.encode_plus를 쓰는 것과의 차이?

# Specify the max length (including both the prompt and the response)
# When calling `generate` with `cache_implementation="static" later, this is also used to create a `StaticCache` object
# with sequence length = `max_length`. The longer the more you will re-use it
model.generation_config.max_length = 128

# StaticCache안쓰고 input_ids 안써도되나

# without `torch.compile`: each call takes ~ 5.0 seconds (on A100 80G + torch 2.3)
# outputs = model.generate(**inputs, do_sample=False)
# response = tokenizer.batch_decode(outputs)[0]
# print(response)

# `torch.compile(model, ...)` is not recommended as you compile callbacks
# and full generate. We recommend compiling only the forward for now. 
# "reduce-overhead" will use cudagraphs. 

# BackendOptions에 device안넣고 그냥 .to(device) 해도 괜찮은건지

model.forward = torch.compile(model.forward, backend=backend, dynamic=False, fullgraph=True, options=options)
model.generation_config.cache_implementation = "static"

# with `torch.compile` (on A100 80G + torch 2.3)
# 1st call: ~ 90 seconds

print("실행 시간 측정 시작")
start_time = time.perf_counter()
outputs = model.generate(**inputs, do_sample=False)
response = tokenizer.batch_decode(outputs)[0]
end_time = time.perf_counter()
print(f"1st call 시간: {end_time - start_time}초")
print(response)


# 2nd call: ~ 60 seconds
start_time = time.perf_counter()
outputs = model.generate(**inputs, do_sample=False)
response = tokenizer.batch_decode(outputs)[0]
end_time = time.perf_counter()
print(f"2nd call 시간: {end_time - start_time}초")
print(response)


# 3nd call: ~ 1.5 seconds
start_time = time.perf_counter()
outputs = model.generate(**inputs, do_sample=False)
response = tokenizer.batch_decode(outputs)[0]
end_time = time.perf_counter()
print(f"3rd call 시간: {end_time - start_time}초")
print(response)
