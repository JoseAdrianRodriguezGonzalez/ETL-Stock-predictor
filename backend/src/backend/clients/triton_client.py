import numpy as np
import tritonclient.http as httpclient

from backend.config import settings


class TritonClient:

    def __init__(self):
        self.client = httpclient.InferenceServerClient(
            url=settings.triton_url,
            verbose=False,
        )

    def is_ready(self) -> bool:
        try:
            return (
                self.client.is_server_live()
                and self.client.is_server_ready()
                and self.client.is_model_ready(
                    settings.triton_model_name,
                    settings.triton_model_version,
                )
            )
        except Exception:
            return False

    def predict(self, inputs: np.ndarray) -> np.ndarray:

        inputs = np.asarray(inputs, dtype=np.float32)

        if inputs.shape != (1, 30, 6):
            raise ValueError(
                f"Expected input shape (1, 30, 6), got {inputs.shape}"
            )

        inference_input = httpclient.InferInput(
            "input",
            inputs.shape,
            "FP32",
        )

        inference_input.set_data_from_numpy(inputs)

        inference_output = httpclient.InferRequestedOutput("output")

        response = self.client.infer(
            model_name=settings.triton_model_name,
            model_version=settings.triton_model_version,
            inputs=[inference_input],
            outputs=[inference_output],
        )

        return response.as_numpy("output")
