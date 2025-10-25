"""Backbone implementation wrapping the Qwen3-VL instruct checkpoint."""

from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
import numpy as np
import torchvision.io as io
import logging


class Backbone:
    """Thin wrapper that provides multimodal query access to Qwen3 models."""

    def __init__(self, model_name: str = "Qwen/Qwen3-VL-8B-Instruct"):
        """Initialise the model weights and paired processor for the chosen checkpoint.

        Parameters
        ----------
        model_name:
            HuggingFace model identifier to load. Defaults to ``Qwen/Qwen3-VL-8B-Instruct``.
        """
        self.model_name = model_name
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_name, dtype="auto", device_map="auto"
        )

        self.processor = AutoProcessor.from_pretrained(model_name)

    def query(
        self,
        visual: np.ndarray | torch.Tensor | None = None,
        text: str | None = None,
        messages: list[dict] | None = None,
    ) -> list[str]:
        """Execute a single-turn or multi-turn conversation with the backbone.

        Parameters
        ----------
        visual:
            Image (H, W, C) or video (T, C, H, W) tensor/array to embed alongside
            ``text`` when ``messages`` is not supplied.
        text:
            Natural-language prompt to pair with ``visual`` when constructing a
            minimal message sequence.
        messages:
            Fully formatted conversation that already follows the processor chat
            template. If provided, ``visual`` and ``text`` are ignored.

        Returns
        -------
        list[str]
            Generated responses corresponding to the supplied conversations.

        Raises
        ------
        ValueError
            Raised when neither ``messages`` nor both ``visual`` and ``text`` are
            provided.
        """
        if messages is None:
            if visual is None or text is None:
                raise ValueError(
                    "Provide either (visual, text) or a prepared messages list."
                )
            content_type = "image" if len(visual.shape) == 3 else "video"
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": content_type,
                            content_type: visual,
                        },
                        {"type": "text", "text": text},
                    ],
                }
            ]
        else:
            logging.debug("Querying model with %d-message conversation.", len(messages))
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        for key, value in list(inputs.items()):
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(next(self.model.parameters()).device)
        generated_ids = self.model.generate(**inputs, max_new_tokens=1024)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_text

    def encode(self, visual: np.ndarray | torch.Tensor):
        """Expose a hook for subclasses to return intermediate embeddings."""
        raise NotImplementedError


if __name__ == "__main__":
    backbone = Backbone()
    image = io.read_image("media/ego-image-speed-limit.png", io.ImageReadMode.RGB)
    print(image.shape)
    answer = backbone.query(image, "What is the speed limit?")
    print(answer)
    vframes, aframes, _ = io.read_video(
        "media/speed-limit-signage.mp4",
        output_format="TCHW",
    )
    print(vframes.shape)
    vframes_cropped = vframes[-128:, ...]
    print(vframes_cropped.shape)
    answer = backbone.query(
        vframes_cropped,
        "Imagine you are the driver of this car. What do you do next? Explain in detail.",
    )
    print(answer[0])
