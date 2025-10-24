from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import torch
import numpy as np
import torchvision.io as io


class Backbone:
    def __init__(self):
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen3-VL-8B-Instruct", dtype="auto", device_map="auto"
        )

        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")

    def query(self, visual: np.ndarray | torch.Tensor, text: str):
        type = "image" if len(visual.shape) == 3 else "video"
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": type,
                        type: visual,
                    },
                    {"type": "text", "text": text},
                ],
            }
        ]
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
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
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
        raise NotImplementedError


if __name__ == "__main__":
    backbone = Backbone()
    image = io.read_image("media/ego-image-speed-limit.png", io.ImageReadMode.RGB)
    print(image.shape)
    answer = backbone.query(image, "What is the speed limit?")
    print(answer)
