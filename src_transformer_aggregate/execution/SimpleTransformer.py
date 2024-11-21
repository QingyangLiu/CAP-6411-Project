import numpy as np
import torch
from timm.models.vision_transformer import VisionTransformer
import torch.nn as nn



class SimpleTransformer(VisionTransformer):

    def __init__(self):
        super().__init__(embed_dim=512, num_heads=8)


    def forward(self, x):

        B=x.shape[0]

        cls_tokens=self.cls_token.expand(B, -1, -1)
        x=torch.cat((cls_tokens, x), dim=1)

        for i, blk in enumerate(self.blocks):
            x=blk(x)
        x=self.norm(x)
        return x[:, 0]




class SimpleTransformer2(VisionTransformer):

    def __init__(self, box_embed_dim=7):
        super().__init__(embed_dim=512, num_heads=8)
        self.box_embed_dim=box_embed_dim
        self.box_cls_token=nn.Parameter(torch.zeros(1,1,box_embed_dim))
        self.box_proj=nn.Linear(box_embed_dim, self.embed_dim)



    def forward(self, x, box):

        B=x.shape[0]

        cls_tokens=self.cls_token.expand(B, -1, -1)
        x=torch.cat((cls_tokens, x), dim=1)
        box_cls_token=self.box_cls_token.expand(B, -1, -1)

        box=torch.cat((box_cls_token, box), dim=1)
        box=self.box_proj(box)
        x=x+box


        for i, blk in enumerate(self.blocks):
            x=blk(x)
        x=self.norm(x)
        return x[:, 0]