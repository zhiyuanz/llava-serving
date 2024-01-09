import io
import json

import numpy as np
from PIL import Image

import triton_python_backend_utils as pb_utils
from transformers import AutoProcessor, LlavaForConditionalGeneration


class TritonPythonModel:
    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        inputs = [
            {"name": "prompt", "data_type": "TYPE_STRING", "dims": [1]},
            {"name": "image",  "data_type": "TYPE_UINT8",  "dims": [-1], "optional": True},
        ]

        outputs = [{"name": "text_output", "data_type": "TYPE_STRING", "dims": [-1]}]

        for input in inputs:
            auto_complete_model_config.add_input(input)
        for output in outputs:
            auto_complete_model_config.add_output(output)

        return auto_complete_model_config

    def initialize(self, args):
        self.model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf").half().cuda()
        self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

    def execute(self, requests):
        prompts = []
        images = []
        for request in requests:
            prompt = pb_utils.get_input_tensor_by_name(request, "prompt").as_numpy()[0]
            prompt = [p.decode("utf-8") for p in prompt]
            prompts += prompt
            image_data = pb_utils.get_input_tensor_by_name(request, "image")
            if image_data is not None:
                image = Image.open(io.BytesIO(image_data.as_numpy().tobytes()))
                images.append(image)
        inputs = self.processor(text=prompts, images=images, return_tensors="pt")
        inputs = {k: v.cuda() if v is not None else None for k, v in inputs.items()}

        generate_ids = self.model.generate(**inputs, max_length=30)
        result = self.processor.tokenizer.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        result = [r.encode("utf-8") for r in result]
        triton_output_tensor = pb_utils.Tensor(
            "text_output", np.asarray(result, dtype=pb_utils.triton_string_to_numpy("TYPE_STRING"))
        )
        return [pb_utils.InferenceResponse(output_tensors=[triton_output_tensor])]