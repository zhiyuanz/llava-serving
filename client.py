#!/usr/bin/env python3

import argparse
import asyncio
import io
import json
import sys
import time

import numpy as np
import tritonclient.grpc.aio as grpcclient
from tritonclient.utils import *
import requests
from transformers import AutoTokenizer

class LLMClient:
    def __init__(self, flags: argparse.Namespace):
        self._client = grpcclient.InferenceServerClient(
            url=flags.url, verbose=flags.verbose
        )
        self._flags = flags
        self._loop = asyncio.get_event_loop()
        self._results_dict = {}
        self._throughput = []

    async def async_request_iterator(self, prompts, sampling_parameters):
        try:
            for iter in range(self._flags.iterations):
                for i, prompt in enumerate(prompts):
                    prompt_id = self._flags.offset + (len(prompts) * iter) + i
                    self._results_dict[str(prompt_id)] = []
                    yield self.create_request(
                        prompt,
                        self._flags.streaming_mode,
                        prompt_id,
                        sampling_parameters,
                    )
        except Exception as error:
            print(f"Caught an error in the request iterator: {error}")

    async def stream_infer(self, prompts, sampling_parameters):
        try:
            # Start streaming
            start_time = time.time()
            response_iterator = self._client.stream_infer(
                inputs_iterator=self.async_request_iterator(
                    prompts, sampling_parameters
                ),
                stream_timeout=self._flags.stream_timeout,
            )
            async for response in response_iterator:
                yield (response, start_time)
        except InferenceServerException as error:
            print(error)
            sys.exit(1)

    async def process_stream(self, prompts, sampling_parameters):
        # Clear results in between process_stream calls
        self.results_dict = []

        # Read response from the stream
        async for response in self.stream_infer(prompts, sampling_parameters):
            result, error = response[0]
            start_time = response[1]
            if error:
                print(f"Encountered error while processing: {error}")
            else:
                output = result.as_numpy("text_output")
                time_elapsed = time.time() - start_time
                for i in output:
                    self._results_dict[result.get_response().id].append(i)
                    self._throughput.append((len(i), time_elapsed))

    async def run(self):
        sampling_parameters = {"temperature": "0.1", "top_p": "0.95"}
        with open(self._flags.input_prompts, "r") as file:
            print(f"Loading inputs from `{self._flags.input_prompts}`...")
            prompts = file.readlines()

        start_time = time.time()
        await self.process_stream(prompts, sampling_parameters)
        print(f"Total time taken: {time.time() - start_time} seconds")
        tokenizer = AutoTokenizer.from_pretrained("liuhaotian/llava-v1.5-7b")
        num_tokens = 0
        for id in self._results_dict.keys():
            for result in self._results_dict[id]:
                result = result.decode('utf-8')[len(prompts[int(id) % len(prompts)]):]
                num_tokens += len(tokenizer.encode(result))
        print("Total tokens generated:", num_tokens)

        with open(self._flags.results_file, "w") as file:
            for id in self._results_dict.keys():
                for result in self._results_dict[id]:
                    file.write(result.decode("utf-8"))
                    file.write("\n")
                file.write("\n=========\n\n")
            print(f"Storing results into `{self._flags.results_file}`...")

        if self._flags.verbose:
            with open(self._flags.results_file, "r") as file:
                print(f"\nContents of `{self._flags.results_file}` ===>")
                print(file.read())

    def run_async(self):
        self._loop.run_until_complete(self.run())

    def create_request(
        self,
        prompt,
        stream,
        request_id,
        sampling_parameters,
        send_parameters_as_tensor=True,
    ):
        inputs = []
        need_batch_dim = not self._flags.model.endswith("vllm")

        try:
            inputs.append(grpcclient.InferInput("text_input", [1, 1] if need_batch_dim else [1], "BYTES"))
            prompt_data = np.array([prompt.encode("utf-8")], dtype=np.object_)
            if need_batch_dim:
                prompt_data = np.expand_dims(prompt_data, axis=0)
            inputs[-1].set_data_from_numpy(prompt_data)
        except Exception as error:
            print(f"Encountered an error during request creation: {error}")


        try:
            response = requests.get('https://www.ilankelman.org/stopsigns/australia.jpg')
            image = np.frombuffer(io.BytesIO(response.content).read(), dtype=np.uint8)  
            if need_batch_dim:
                image = np.expand_dims(image, axis=0)
            inputs.append(grpcclient.InferInput("image", image.shape, "UINT8"))
            inputs[-1].set_data_from_numpy(image)
        except Exception as error:
            print(f"Encountered an error during request creation: {error}")

        if self._flags.model.endswith("vllm"):
            stream_data = np.array([stream], dtype=bool)
            inputs.append(grpcclient.InferInput("stream", [1], "BOOL"))
            inputs[-1].set_data_from_numpy(stream_data)

            # Request parameters are not yet supported via BLS. Provide an
            # optional mechanism to send serialized parameters as an input
            # tensor until support is added

            if send_parameters_as_tensor:
                sampling_parameters_data = np.array(
                    [json.dumps(sampling_parameters).encode("utf-8")], dtype=np.object_
                )
                inputs.append(grpcclient.InferInput("sampling_parameters", [1], "BYTES"))
                inputs[-1].set_data_from_numpy(sampling_parameters_data)


        # Add requested outputs
        outputs = []
        outputs.append(grpcclient.InferRequestedOutput("text_output"))

        # Issue the asynchronous sequence inference.
        return {
            "model_name": self._flags.model,
            "inputs": inputs,
            "outputs": outputs,
            "request_id": str(request_id),
            "parameters": sampling_parameters,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        required=False,
        default="vllm_model",
        help="Model name",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        required=False,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "-u",
        "--url",
        type=str,
        required=False,
        default="localhost:8001",
        help="Inference server URL and its gRPC port. Default is localhost:8001.",
    )
    parser.add_argument(
        "-t",
        "--stream-timeout",
        type=float,
        required=False,
        default=None,
        help="Stream timeout in seconds. Default is None.",
    )
    parser.add_argument(
        "--offset",
        type=int,
        required=False,
        default=0,
        help="Add offset to request IDs used",
    )
    parser.add_argument(
        "--input-prompts",
        type=str,
        required=False,
        default="prompts.txt",
        help="Text file with input prompts",
    )
    parser.add_argument(
        "--results-file",
        type=str,
        required=False,
        default="results.txt",
        help="The file with output results",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        required=False,
        default=1,
        help="Number of iterations through the prompts file",
    )
    parser.add_argument(
        "-s",
        "--streaming-mode",
        action="store_true",
        required=False,
        default=False,
        help="Enable streaming mode",
    )
    FLAGS = parser.parse_args()

    client = LLMClient(FLAGS)
    client.run_async()
