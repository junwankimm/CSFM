import torch.nn as nn
from transformers import CLIPTokenizer, CLIPTextModel
from transformers import AutoTokenizer, Siglip2TextModel
import time


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z, {'token_embedding': outputs.last_hidden_state, 'pooler_output': outputs.pooler_output, 'token_mask': batch_encoding['attention_mask'].to(self.device), 'tokens': batch_encoding["input_ids"].to(self.device)}

    def encode_from_token(self, tokens):
        tokens = tokens.to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z
    
    def encode(self, text):
        return self(text)


class FrozenCLIPTokenizer(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="openai/clip-vit-large-patch14", device="cuda", max_length=77):
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"]
        return tokens

    def encode(self, text):
        return self(text)
    
class FrozenSigLIP2Embedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from Hugging Face)"""
    def __init__(self, version="google/siglip2-base-patch16-256", device="cuda", max_length=64):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(version)
        self.transformer = Siglip2TextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.freeze()

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        batch_encoding = self.tokenizer(text, truncation=True, max_length=self.max_length, return_length=True,
                                        return_overflowing_tokens=False, padding="max_length", return_tensors="pt")
        tokens = batch_encoding["input_ids"].to(self.device)
        token_masks = tokens > 0
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z, {'token_embedding': outputs.last_hidden_state, 'pooler_output': outputs.pooler_output, 'token_mask': token_masks}

    def encode_from_token(self, tokens):
        tokens = tokens.to(self.device)
        outputs = self.transformer(input_ids=tokens)

        z = outputs.last_hidden_state
        return z
    
    def encode(self, text):
        return self(text)
    
if __name__ == "__main__":
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text = ["A toucan with a bright yellow chest, black body, and a large, colorful beak featuring green, yellow, and red hues perches on a thick tree branch. Behind it, another toucan with similar coloring is visible on a higher branch. The setting is a lush, dense jungle environment with green foliage, ferns, palm fronds, and tree trunks. A wooden feeding platform holds a metal dish containing red and white food. The scene is viewed through vertical bars, suggesting an enclosure."]
    model = FrozenCLIPEmbedder(device=device).to(device)
    # text = ["A photo of a cat", "A photo of a dog"]
    breakpoint()
    
    z, info = model.encode(text)
    end = time.time()
    print("Time taken:", end - start)
    print("Encoded shape:", z.shape)
