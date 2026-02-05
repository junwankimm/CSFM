from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn as nn

str_to_dtype = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError
    
# Tested with Qwen3-0.6B meta-llama/Llama-3.2-1B
class FrozenLLMEmbedder(AbstractEncoder):
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B", max_length=256, model_dtype="bfloat16", output_dtype="float32", device="cuda"):
        super().__init__()
        model_dtype = str_to_dtype.get(model_dtype, torch.bfloat16)
        output_dtype = str_to_dtype.get(output_dtype, torch.float32)
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=model_dtype,
        )
        self.device = device # For legacy support
        self.output_dtype = output_dtype
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length
        for param in self.model.parameters():
            param.requires_grad = False
    
    def encode(self, prompt_batch):
        with torch.no_grad():
            text_inputs = self.tokenizer(
                prompt_batch,
                padding="max_length", # For TPU Support
                max_length=self.max_length,
                truncation=True,
                return_tensors="pt",
            ).to(self.model.device)
        
        text_input_ids = text_inputs.input_ids
        prompt_masks = text_inputs.attention_mask

        prompt_embeds = self.model(
            input_ids=text_input_ids,
            attention_mask=prompt_masks,
            output_hidden_states=True,
        ).hidden_states[-2].to(self.output_dtype)
        
        latents_and_others = {
            "token_embedding": prompt_embeds,
            "pooler_output": None,
            "token_mask": prompt_masks.to(self.output_dtype),
            "token_ids": text_input_ids.to(self.output_dtype),
        }
        
        return prompt_embeds, latents_and_others

if __name__ == "__main__":
    encoder = FrozenLLMEmbedder().to("cuda").eval()
    breakpoint()
    prompts = [
        "The image depicts a person in military attire, including a helmet and tactical gear, running through a grassy field dotted with yellow flowers. The individual appears to be in motion, possibly during a training exercise or a tactical operation. The background shows a mix of greenery and a dirt path, suggesting an open outdoor setting. The person's posture and the way they are holding their hands up might indicate a gesture of surrender or signaling. The overall scene conveys a sense of action and movement in a natural environment.",
        "The image captures a picturesque coastal scene with a clear blue sky and scattered clouds. In the foreground, a calm body of water reflects the serene atmosphere. A dock extends into the water, with a flagpole and a small structure at its end. Beyond the dock, a marina is visible, filled with various boats and yachts moored along the pier. The midground features a charming town with white buildings adorned with red roofs, and a prominent bell tower rises above the skyline. In the background, a range of rugged mountains provides a dramatic backdrop, adding depth and grandeur to the landscape. The overall composition exudes tranquility and natural beauty, typical of a Mediterranean coastal town. The image captures a picturesque coastal scene with a clear blue sky and scattered clouds. In the foreground, a calm body of water reflects the serene atmosphere. A dock extends into the water, with a flagpole and a small structure at its end. Beyond the dock, a marina is visible, filled with various boats and yachts moored along the pier. The midground features a charming town with white buildings adorned with red roofs, and a prominent bell tower rises above the skyline. In the background, a range of rugged mountains provides a dramatic backdrop, adding depth and grandeur to the landscape. The overall composition exudes tranquility and natural beauty, typical of a Mediterranean coastal town. The image captures a picturesque coastal scene with a clear blue sky and scattered clouds. In the foreground, a calm body of water reflects the serene atmosphere. A dock extends into the water, with a flagpole and a small structure at its end. Beyond the dock, a marina is visible, filled with various boats and yachts moored along the pier. The midground features a charming town with white buildings adorned with red roofs, and a prominent bell tower rises above the skyline. In the background, a range of rugged mountains provides a dramatic backdrop, adding depth and grandeur to the landscape. The overall composition exudes tranquility and natural beauty, typical of a Mediterranean coastal town.",
        "설마 NaN 뜨는게 영어아닌게 들어와서인가",
        ""]
    embeds, others = encoder.encode(prompts)
    print(embeds.shape)

    
    encoder = FrozenLLMEmbedder(model_name="Qwen/Qwen3-0.6B", max_length=128).to("cuda").eval()
    embeds, others = encoder.encode(prompts)
    print(embeds.shape)
    
    