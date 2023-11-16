import os
import torch
import tensorrt as trt
from transformers import CLIPVisionModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Build clip encoder engine
vision_tower_name = "openai/clip-vit-large-patch14-336"
vision_tower = CLIPVisionModel.from_pretrained(vision_tower_name).half().to(device)
random_input = torch.randn(1, 3, 336, 336).half().to(device)
output = vision_tower(random_input)
assert output.last_hidden_state.shape == (1, 577, 1024)

print("Saving PyTorch CLIP model...")
torch.save(vision_tower.state_dict(), "clip.pt")
print("Loading PyTorch CLIP model...")
vision_tower = torch.load("clip.pt")
print("Saving ONNX CLIP model...")
torch.onnx.export(
    vision_tower,
    random_input,
    'clip.onnx',
    opset_version=17,
    input_names=['input'],
    output_names=['output']
)

trt_logger = trt.Logger(trt.Logger.ERROR)
builder = trt.Builder(trt_logger)
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(network, logger)
with open('clip.onnx', 'rb') as model:
    print("Parsing model")
    parser.parse(model.read())
    print("Found {} errors".format(parser.num_errors))
    for i in range(parser.num_errors):
        print(parser.get_error(i))
    print("Succeeded parsing %s" % 'clip..onnx')

# profile = builder.create_optimization_profile()
config = builder.create_builder_config()

# input_tensor = network.get_input(0)
# input_tensor.shape = [-1, 3, 336, 336]
# profile.set_shape(input_tensor.name, min=(1, 3, 336, 336), opt=(1, 3, 336, 336), max=(1, 3, 336, 336))
# config.add_optimization_profile(profile)

from time import time
t0 = time()
engineString = builder.build_serialized_network(network, config)
t1 = time()
if engineString == None:
    print("Failed building TensorRT engine!")
else:
    print("Succeeded building %s in %d s" % ('clip.trt', t1 - t0))

with open('clip.trt', 'wb') as f:
    f.write(engineString)

# 2. Build mm projector engine
# 3. Build llama engine
