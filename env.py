import random
import torch
from PIL import Image
import open_clip
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import requests
from io import BytesIO

class Env():
    def __init__(self, images, device='cpu'):
        self.images = np.array(images)
        print(len(self.images))
        self.seen_images = [] 
        self.has_not_seen_images = list(range(len(images)))  # Initialize with all image indices

        self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
        self.model = self.model.to(device)
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

        self.current_images = None
        self.device = device

    def reset(self):
        self.seen_images = []
        self.has_not_seen_images = list(range(len(self.images)))
        self.sample(1)

        return self.current_images

    def sample(self, num_images):
        # randomly generating an index from the images that have not been seen by the agent
        sampled_indices = random.sample(self.has_not_seen_images, num_images)
        # removing the indicies from has_not_seen
        self.has_not_seen_images = [idx for idx in self.has_not_seen_images if idx not in sampled_indices]
        
        self.current_images = self.images[np.array(sampled_indices)][0]

        self.current_images = self.image_preprocess(self.current_images)

    def step(self, caption):
        reward = self.reward(caption)

        # if agent has seen all the images
        if len(self.has_not_seen_images) == 0:
            done = True
            self.reset()
        else:
            done = False
            self.sample(1)

        return self.current_images, reward, done
        
    def reward(self, caption):
        # image = self.preprocess(self.current_images).unsqueeze(0)

        image = self.current_images.unsqueeze(0)

        text = self.tokenizer([caption])

        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
    
        sim_score = image_features @ text_features.T

        return sim_score
        # some reward calculation

    def image_preprocess(self, image):
        # Define transformations for preprocessing the image
        preprocess_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Load the image from the given URL or path
        if image.startswith('http'):
            # If image is from URL, download and open
            response = requests.get(image)
            image = Image.open(BytesIO(response.content))
        else:
            # If image is from local path, open directly
            image = Image.open(image)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply preprocessing transformations
        image_tensor = preprocess_transform(image)
        
        return image_tensor.to(self.device)

