import sys

import tensorrt as trt
import torch
from tensorrt_llm.runtime import Session, TensorInfo

def load_and_run_trt_engine(input_tensor, engine_path, output_name='output'):
    with open(engine_path, 'rb') as f:
        engine_buffer = f.read()
    session = Session.from_serialized_engine(engine_buffer)
    output_info = session.infer_shapes(
        [TensorInfo('input', trt.DataType.HALF, input_tensor.shape)])
    outputs = {
        t.name: torch.empty(tuple(t.shape), dtype=torch.float16, device='cuda')
        for t in output_info
    }
    session.run(
        {
            'input': input_tensor
        },
        outputs,
        torch.cuda.current_stream().cuda_stream
    )

    return outputs[output_name]

fake_image_tensor = torch.ones((1, 3, 336, 336), dtype=torch.float16, device='cuda')
# Choose the second last layer as the image feature
image_feature = load_and_run_trt_engine(fake_image_tensor.contiguous(), './clip.trt', 'input.188')  # TODO: name hidden layers
image_feature = image_feature[:, 1:]
image_feature = load_and_run_trt_engine(image_feature, './mm_projector.trt')
print(image_feature)