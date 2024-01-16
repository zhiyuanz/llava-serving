# LLaVa Serving

Prototypes of the LLaVa model served on the Triton Inference Server with different backends

## Usage

```bash
# Start the Triton Inference Server container 
docker run --rm -it --net host --shm-size=2g --ulimit memlock=-1 --ulimit stack=67108864 --gpus all -v ${PWD}:code -w /code nvcr.io/nvidia/tritonserver:23.12-trtllm-python-py3 bash
```
### vLLM backend
```bash
# Pickup the devel branch of vLLM with experimental LLaVa support
# https://github.com/vllm-project/vllm/pull/2153
pip install git+https://github.com/zhiyuanz/vllm.git@llava_devel

tritonserver --model-repository ./model_repository --model-control-mode=explicit --load-model=llava_vllm
```

### Python backend (🤗 Transformers)
```bash
# Install the recent transformers with LLaVa support
pip install transformers==4.36.1

tritonserver --model-repository ./model_repository --model-control-mode=explicit --load-model=llava_py
```

### TensorRT-LLM backend
1. Build the TensorRT-LLM container by following the [instruction](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/installation.md#option-1-build-tensorrt-llm-in-one-step).

2. Inside the TensorRT-LLM container, run scripts to build the TensorRT engines
```bash
# Build engine for the visual tower
python ./llava-serving/build.py

# Download the model weights manually for using the TensorRT-LLM's build script
export MODEL_NAME="llava-v1.5-7b"
git clone https://huggingface.co/liuhaotian/llava-v1.5-7b tmp/hf_models/llava-v1.5-7b

# Build engine for the language model
cd ./example/llama
python build.py \
  --model_dir tmp/hf_models/llava-v1.5-7b \
  --output_dir trt_engines/llava-v1.5-7b/fp16/1-gpu \
  --dtype float16 \
  --remove_input_padding \
  --use_gpt_attention_plugin float16 \
  --enable_context_fmha \
  --use_gemm_plugin float16 \
  --max_prompt_embedding_table_size 576 \
  --max_batch_size 1 \
  --use_inflight_batching
```

3. Copy the TensorRT engine file to `model_repository`
```bash
cp .../trt_engines/llava-v1.5-7b/fp16/1-gpu/* ./model_repository/llava_trt_llm/
cp .../visual_engines/llava-v1.5-7b/visual_encoder_fp16.engine ./model_repository/llava_trt_vision/
```

4. Run the server with the model ensemble
```bash
tritonserver --model-repository ./model_repository --model-control-mode=explicit \
  --load-model=llava_trt_vision --load-model=llava_trt_llm --load-model=llava_trt_ensemble
```

## Benchmark
`client.py` is an example client which runs a simple benchmark with a hardcoded image + a list of prompts from file, and records the total number of tokens generated.

As reference, on a single A10G card I got the following average throughput with some naive prompts and the defult parameters:

| Model | Average Throughput
| :----------------------: | :-----------------------------: |
| Python | 23 tokens/sec |
| vLLM | 46 tokens/sec |
| TensorRT-LLM | 41 tokens/sec |
