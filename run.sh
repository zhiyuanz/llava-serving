# Install transformers' main branch with llava support
# https://huggingface.co/docs/transformers/main/model_doc/llava
pip install git+https://github.com/huggingface/transformers.git


# Pick the llava_devel branch of vllm
# https://github.com/AzureSilent/vllm/tree/llava_devel
# https://github.com/vllm-project/vllm/pull/2153
pip install git+https://github.com/AzureSilent/vllm.git@llava_devel

# Start Triton Server
tritonserver \
    --model-repository ./model_repository \
    --model-control-mode=explicit \
    --load-model=llava_hf