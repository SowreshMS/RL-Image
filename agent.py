import torch
from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel
from transformers import GPT2Tokenizer, GPT2Model

import torch
import torch.nn as nn
import torch.nn.functional as F


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.vit = self.clip.vision_model
        self.visual_projection = self.clip.visual_projection
        self.clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

        self.gpt = GPT2Model.from_pretrained('gpt2')
        self.gpt_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.out = nn.Linear(768, 50257)

    def forward(self, image):
        # image should be tensor with shape (1, 3, 224, 224)
        image = image.unsqueeze(0)

        with torch.no_grad(), torch.cuda.amp.autocast():
            x = self.vit(image)[0]
            x = self.visual_projection(x)
        
        for block in self.gpt.h:
            x = block(x)[0]
        
        x = self.gpt.ln_f(x)

        x = self.out(x)

        return x, self.get_caption(x)


    def get_caption(self, text):
        if len(text.shape) == 3:
            text = text.squeeze(0)

        # Apply softmax along the token dimension (axis 1)
        probabilities = F.softmax(text, dim=1)

        # Sample a token using the probabilities
        sampled_token_index = torch.multinomial(probabilities, num_samples=1)

        # Convert the sampled index back to a tensor
        sampled_token = torch.tensor(sampled_token_index)

        string = [self.gpt_tokenizer.decode(token.item()) for token in sampled_token]
        
        return ' '.join(string)
          
        