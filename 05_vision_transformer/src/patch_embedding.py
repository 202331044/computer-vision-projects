import torch.nn as nn
import torch


# class _PatchEmbedding(nn.Module):
#     def __init__(self, patch_size, patch_dim, d_model):
#         super().__init__()
#         self.patch_size = patch_size
#         self.fc = nn.Linear(patch_dim, d_model)
    
#     def forward(self, x):
#         B, C, _, _ = x.shape[:]

#         patches = x.unfold(2, self.patch_size, self.patch_size)
#         patches = patches.unfold(3, self.patch_size, self.patch_size)

#         patches = patches.permute(0, 2, 3, 1, 4, 5)

#         patches = patches.reshape(B, -1, C * self.patch_size * self.patch_size)
    
#         output = self.fc(patches)

#         return output


class PatchEmbedding(nn.Module):
    def __init__(self, in_channel, patch_size, d_model):
        super().__init__()
        self.cnn = nn.Conv2d(in_channels=in_channel,
                          out_channels=d_model,
                          kernel_size=patch_size,
                          stride=patch_size)

    def forward(self, x):
        x = self.cnn(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)

        return x