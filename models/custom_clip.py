
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
from torch.nn.modules.utils import _pair
from torch.nn import Conv2d, Dropout
import clip

from functools import reduce
from operator import mul
import numpy as np
from PIL import Image
import math

class MaskedTextTransformer(nn.Module):
    def __init__(self, clip_model, blocks_to_mask):
        super().__init__()
        """
        clip_model: CLIP model
        blocks_to_mask: list of block indices to mask (0-11)
        """

        self.text = clip_model.text  # text backbone from CLIP
        self.dtype = clip_model.dtype

        # No gradients for the clip model parameters
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection

        # Infer the feature size from the transformer
        # This assumes all blocks output features of the same size
        feature_size = self.text.transformer.width

        print("Feature size:", feature_size)

        # Initialize masks for each specified block
        self.masks = nn.ParameterDict({
            str(block): nn.Parameter(torch.ones(50, 1, feature_size)) for block in blocks_to_mask
        })

    def embed_text(self, x):
        """
        The input is text
        """
        x = self.token_embedding(x)
        x = x + self.positional_embedding.to(x.dtype)

        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        
        return x

    def encoder(self, x):
        x = x.permute(1, 0, 2)  # NLD -> LND

        # Process each encoder block and apply mask if the block is in the mask list
        for i, block in enumerate(self.transformer.resblocks):
            x = block(x)
            if str(i) in self.masks:
                x = x * self.masks[str(i)]

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def forward(self, text_input):
        x = self.embed_text(text_input.type(self.dtype))
        x = self.encoder(x)
        return x

class MaskedVisualTransformer(nn.Module):
    def __init__(self, clip_model, blocks_to_mask):
        super().__init__()
        """
        clip_model: CLIP model
        blocks_to_mask: list of block indices to mask (0-11)
        """

        self.visual = clip_model.visual  # visual backbone from CLIP
        self.dtype = clip_model.dtype

        # Turn off all gradients for all the clip model parameters
        for param in self.visual.parameters():
            param.requires_grad = False
        
        # Infer the feature size from the transformer
        # This assumes all blocks output features of the same size
        feature_size = self.visual.transformer.width

        # Initialize masks for each specified block and ensure they are in the datatype of the visual transformer
        self.masks = nn.ParameterDict({
            str(block): nn.Parameter(torch.zeros(50, 1, feature_size).type(self.dtype)) for block in blocks_to_mask
        })

        # self.mask = nn.Parameter(torch.zeros(1, 512).type(self.dtype)).requires_grad_(True)

    def embed_image(self, x):
        """
        The input is tensor image
        """
        x = self.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual.positional_embedding.to(x.dtype)
        return x
    
    def encoder(self, x):
        x = self.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        # Process each encoder block and apply mask if the block is in the mask list
        for i, block in enumerate(self.visual.transformer.resblocks):
            x = block(x)
            if str(i) in self.masks:
                x = x * self.masks[str(i)]

        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.visual.ln_post(x[:, 0, :])

        if self.visual.proj is not None:
            x = x @ self.visual.proj

        # x = x * self.mask
        return x

    def forward(self, image_input):
        x = self.embed_image(image_input.type(self.dtype))
        x = self.encoder(x)
        return x

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    clip_model, preprocess = clip.load("ViT-B/32", device=device)    
    clip_model.eval()

    masked_clip_model = MaskedVisualTransformer(clip_model, blocks_to_mask=[])
    masked_clip_model.eval()

    # for name, param in masked_clip_model.named_parameters():
    #     if 'mask' in name:
    #         print(name, param.shape)
    # assert False
    # Load and preprocess the image
    image = preprocess(Image.open("models/donkey.jpg")).unsqueeze(0).to(device)
    text = clip.tokenize(["a photo of a donkey", "a photo of a horse"]).to(device)
    with torch.no_grad():

        image_features_mask = masked_clip_model(image)

        image_features = CLIP_image_encoder(image)

        prompted_image_features = visual_prompter(image)
        prompted_text_features = text_prompter(["a photo of a donkey", "a photo of a horse"])

        # Compare the text and image features
        logits_per_image, logits_per_text = clip_model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    print("Label probs:", probs)  
    # Cos similarity between the image and text features
    print("Cosine similarity:", F.cosine_similarity(image_features, prompted_image_features).cpu().numpy())
    print("Cosine similarity:", F.cosine_similarity(image_features, image_features_mask).cpu().numpy())