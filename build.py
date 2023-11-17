import os
import torch
import tensorrt as trt
from transformers import CLIPVisionModel
from time import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. Build clip encoder engine
def build_vision_tower():
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
    parser = trt.OnnxParser(network, trt_logger)
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
from huggingface_hub import hf_hub_download
from tensorrt_llm.module import Module
from tensorrt_llm.layers import Linear
from tensorrt_llm.functional import gelu
from tensorrt_llm import net_guard
from tensorrt_llm.functional import Tensor
from tensorrt_llm import Builder
import tensorrt_llm

def build_mm_projector():
    hf_hub_download("liuhaotian/llava-v1.5-7b", "mm_projector.bin", local_dir="./")
    mm_projector_weights = torch.load('mm_projector.bin', map_location='cpu')

    class MMProjector(Module):
        def __init__(self):
            super().__init__()
            self.proj1 = Linear(1024, 4096, bias=True, dtype='float16')
            self.proj2 = Linear(4096, 4096, bias=True, dtype='float16')

        def forward(self, x):
            out = self.proj1(x)
            out = gelu(out)
            out = self.proj2(out)
            out.mark_output('output', out.dtype)
            return out
    
    mm_projector = MMProjector()
    print("Binding weights")
    mm_projector.proj1.weight.value = mm_projector_weights['model.mm_projector.0.weight'].numpy()
    mm_projector.proj1.bias.value = mm_projector_weights['model.mm_projector.0.bias'].numpy()
    mm_projector.proj2.weight.value = mm_projector_weights['model.mm_projector.2.weight'].numpy()
    mm_projector.proj2.bias.value = mm_projector_weights['model.mm_projector.2.bias'].numpy()

    builder = Builder()
    builder_config = builder.create_builder_config(
        precision=trt.float16,
        strongly_typed=True,
    )
    network = builder.create_network()
    network.trt_network.name = "mm_projector"

    with net_guard(network):
        network.set_named_parameters(mm_projector.named_parameters())
        hidden_states = Tensor(
            name='input',
            dtype=trt.float16,
            shape=[1, 577, 1024])
        output = mm_projector(hidden_states)

    tensorrt_llm.graph_rewriting.optimize(network)
    t0 = time()
    engine = builder.build_engine(network, builder_config)
    t1 = time()
    if engine == None:
        print("Failed building TensorRT engine!")
    else:
        print("Succeeded building %s in %d s" % ('./mm_projector.trt', t1 - t0))

    with open('./mm_projector.trt', 'wb') as f:
        f.write(engine)

# 3. Build llama engine
