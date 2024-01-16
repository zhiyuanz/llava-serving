import argparse
import os
from time import time

from PIL import Image
import tensorrt as trt
import torch
import tensorrt_llm
from transformers import CLIPVisionModel

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
def build_mm_projector():
    from huggingface_hub import hf_hub_download
    from tensorrt_llm import Builder, net_guard
    from tensorrt_llm.functional import Tensor, gelu
    from tensorrt_llm.layers import Linear
    from tensorrt_llm.module import Module
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
def build_llama():
    from llava.mm_utils import get_model_name_from_path
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from tensorrt_llm.examples.llama.weight import load_from_hf_llama
    from tensorrt_llm.layers.attention import PositionEmbeddingType
    from tensorrt_llm.quantization import QuantMode
    disable_torch_init() # Disable the redundant torch default initialization to accelerate model creation.
    model_path = "liuhaotian/llava-v1.5-7b"
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, device="cpu")
    llama_config = model.config
    tensorrt_llm_llama = tensorrt_llm.models.LLaMAForCausalLM(
            num_layers=llama_config.num_hidden_layers,
            num_heads=llama_config.num_attention_heads,
            num_kv_heads=llama_config.num_key_value_heads,
            hidden_size=llama_config.hidden_size,
            vocab_size=llama_config.vocab_size,
            hidden_act=llama_config.hidden_act,
            max_position_embeddings=llama_config.max_position_embeddings,
            dtype='float16',
            mlp_hidden_size=llama_config.intermediate_size,
            position_embedding_type=PositionEmbeddingType.rope_gpt_neox,
            rotary_base=10000.0,
            rotary_scaling=None,
            use_parallel_embedding=False,
            embedding_sharding_dim=1,
            quant_mode=QuantMode(0),
            rms_norm_eps=llama_config.rms_norm_eps)
    load_from_hf_llama(tensorrt_llm_llama, model, dtype='float16')


# Build the visual tower in one step
def build_trt_engine(img_height,
                     img_width,
                     output_dir,
                     minBS=1,
                     optBS=2,
                     maxBS=4):
    part_name = 'visual_encoder'
    onnx_file = '%s/%s.onnx' % (output_dir, part_name)
    engine_file = '%s/%s_fp16.engine' % (output_dir, part_name)

    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    profile = builder.create_optimization_profile()
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)

    parser = trt.OnnxParser(network, logger)

    with open(onnx_file, 'rb') as model:
        if not parser.parse(model.read(), "/".join(onnx_file.split("/"))):
            logger.log(trt.Logger.ERROR, "Failed parsing %s" % onnx_file)
            for error in range(parser.num_errors):
                logger.log(trt.Logger.ERROR, parser.get_error(error))
        logger.log(trt.Logger.INFO, "Succeeded parsing %s" % onnx_file)

    nBS = -1
    nMinBS = minBS
    nOptBS = optBS
    nMaxBS = maxBS

    H, W = img_height, img_width
    inputT = network.get_input(0)
    inputT.shape = [nBS, 3, H, W]
    profile.set_shape(inputT.name, [nMinBS, 3, H, W], [nOptBS, 3, H, W],
                        [nMaxBS, 3, H, W])

    config.add_optimization_profile(profile)

    t0 = time()
    engine_string = builder.build_serialized_network(network, config)
    t1 = time()
    if engine_string is None:
        raise RuntimeError("Failed building %s" % (engine_file))
    else:
        logger.log(trt.Logger.INFO,
                   "Succeeded building %s in %d s" % (engine_file, t1 - t0))
        with open(engine_file, 'wb') as f:
            f.write(engine_string)

def build_llava_engine(args):
    from transformers import AutoProcessor, LlavaForConditionalGeneration
    image_processor = AutoProcessor.from_pretrained(args.model_name)
    model = LlavaForConditionalGeneration.from_pretrained(args.model_name)

    image = Image.new('RGB', [10, 10])  # dummy image
    image = image_processor(text="placeholder", images=image, return_tensors='pt')['pixel_values']
    image = image.half().cuda()

    class MultiModalProjector(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.vision_tower = model.vision_tower.to('cuda')
            self.projector = model.multi_modal_projector.to('cuda')

        def forward(self, image):
            image_outputs = self.vision_tower(image, output_hidden_states=True)
            vision_feature_layer = -2
            selected_image_feature = image_outputs.hidden_states[vision_feature_layer]
            return self.projector(selected_image_feature)
            
    visual_wrapper = MultiModalProjector()
    torch.onnx.export(visual_wrapper,
                      image,
                      f'{args.output_dir}/visual_encoder.onnx',
                      opset_version=17,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {
                          0: 'batch'
                      }})

    build_trt_engine(image.shape[2], image.shape[3], args.output_dir)

if __name__ == '__main__':
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    logger = trt.Logger(trt.Logger.ERROR)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name',
                        type=str,
                        default='llava-hf/llava-1.5-7b-hf',
                        help="Model name")
    parser.add_argument('--output_dir',
                        type=str,
                        default='visual_engines',
                        help="Directory where visual TRT engines are saved")
    args = parser.parse_args()

    args.output_dir = args.output_dir + "/" + args.model_name.split("/")[1]
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    build_llava_engine(args)
