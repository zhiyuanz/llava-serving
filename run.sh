# Start Triton Server
tritonserver \
    --model-repository ./model_repository \
    --model-control-mode=explicit \
    --load-model=llava_hf