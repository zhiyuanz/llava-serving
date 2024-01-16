import io

from PIL import Image
import numpy as np
import tensorrt as trt
import torch
from transformers import AutoProcessor
import triton_python_backend_utils as pb_utils

from tensorrt_llm.runtime import Session, TensorInfo
import tensorrt_llm

# TODO: parameterize this
REQUEST_OUTPUT_LEN = 100
PROMPT_EMBDDING_TABLE_SIZE = 576

class TritonPythonModel:

    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        inputs = [
            {"name": "text_input", "data_type": "TYPE_STRING", "dims": [1]},
            {"name": "image",  "data_type": "TYPE_UINT8",  "dims": [-1], "optional": True},
        ]

        outputs = [
            {"name": "input_ids", "data_type": "TYPE_INT32", "dims": [-1]},
            {"name": "input_lengths", "data_type": "TYPE_INT32", "dims": [1]},
            {"name": "request_output_len", "data_type": "TYPE_INT32", "dims": [1]},
            {"name": "prompt_vocab_size", "data_type": "TYPE_INT32", "dims": [1]},
            {"name": "prompt_embedding_table", "data_type": "TYPE_FP16", "dims": [-1, -1]},
        ]

        for input in inputs:
            auto_complete_model_config.add_input(input)
        for output in outputs:
            auto_complete_model_config.add_output(output)

        return auto_complete_model_config

    def initialize(self, args):
        self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        vit_path = "/code/model_repository/llava_trt_vision/visual_encoder_fp16.engine"
        with open(vit_path, 'rb') as f:
            engine_buffer = f.read()
        self.vit_session = Session.from_serialized_engine(engine_buffer)

    def execute(self, requests):
        responses = []
        for _, request in enumerate(requests):
            # Get input tensors
            prompt = pb_utils.get_input_tensor_by_name(request, 'text_input').as_numpy()[0]
            prompt = [p.decode('utf-8') for p in prompt]
            image_data = pb_utils.get_input_tensor_by_name(request, "image")
            images = []
            if image_data is not None:
                image = Image.open(io.BytesIO(image_data.as_numpy().tobytes()))
                images = [image]
            inputs = self.processor(text=prompt, images=images, return_tensors="pt")
            input_ids = inputs["input_ids"]
            pixel_values = inputs["pixel_values"]

            visual_features = {
                'input': pixel_values.half().to('cuda')
            }
            visual_output_info = self.vit_session.infer_shapes(
                [TensorInfo('input', trt.DataType.HALF, pixel_values.shape)])
            visual_outputs = {
                t.name: torch.empty(tuple(t.shape),
                                    dtype=torch.float16,
                                    device="cuda")
                for t in visual_output_info
            }

            self.vit_session.run(visual_features, visual_outputs, torch.cuda.current_stream().cuda_stream)
            visual_embedding = visual_outputs['output'].cpu().detach()

            input_ids, prompt_table = self.setup_fake_prompts(visual_embedding, input_ids)
            prompt_table = np.expand_dims(prompt_table, axis=0)
            
            input_id_tensor = pb_utils.Tensor(
                'input_ids', input_ids.numpy())
            input_lengths_tensor = pb_utils.Tensor(
                'input_lengths', torch.tensor([input_ids.shape[1]]).to(torch.int32).unsqueeze(0).numpy())
            request_output_len_tensor = pb_utils.Tensor(
                'request_output_len', torch.tensor([REQUEST_OUTPUT_LEN]).to(torch.int32).unsqueeze(0).numpy())
            prompt_embedding_table_tensor = pb_utils.Tensor(
                'prompt_embedding_table', prompt_table)
            prompt_vocab_size_tensor = pb_utils.Tensor(
                'prompt_vocab_size', torch.tensor([PROMPT_EMBDDING_TABLE_SIZE]).to(torch.int32).unsqueeze(0).numpy())

            inference_response = pb_utils.InferenceResponse(output_tensors=[
                input_id_tensor, 
                input_lengths_tensor,
                request_output_len_tensor,
                prompt_embedding_table_tensor,
                prompt_vocab_size_tensor,
            ])
            responses.append(inference_response)
        return responses

    def setup_fake_prompts(self, visual_features, input_ids):
        # Assemble fake prompts which points to image embedding actually
        vocab_size = self.processor.tokenizer.vocab_size
        fake_prompt_id = torch.arange(vocab_size,
            vocab_size + visual_features.shape[0] * visual_features.shape[1])
        fake_prompt_id = fake_prompt_id.reshape(visual_features.shape[0],
                                                visual_features.shape[1])

        input_ids = [fake_prompt_id, input_ids]
        input_ids = torch.cat(input_ids,
                              dim=1).contiguous().to(torch.int32)
        
        prompt_table = visual_features.view(
            (visual_features.shape[0] * visual_features.shape[1],
                visual_features.shape[2])).to(torch.float16)
        prompt_table = prompt_table[1:, :] # remove the cls token
        
        return input_ids, prompt_table
