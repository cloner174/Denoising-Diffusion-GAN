import torch
import torch.nn as nn


class DiscriminatorFeatureExtractor(nn.Module):
    
    def __init__(self, D_small):
        """
        D_small: an instance of Discriminator_small.
        """
        super().__init__()
        
        self.t_embed = D_small.t_embed
        self.act = D_small.act
        
        self.start_conv = D_small.start_conv
        self.conv1 = D_small.conv1
        self.conv2 = D_small.conv2
        self.conv3 = D_small.conv3
        self.conv4 = D_small.conv4
        
        self.stddev_group = D_small.stddev_group
        self.stddev_feat = D_small.stddev_feat
        
        self.final_conv = D_small.final_conv
    
    
    def forward(self, x, t, x_t):
        """
        x:   a batch of images (e.g., shape [B, 3, H, W])
        t:   timesteps (or conditioning), shape [B] or [B, 1]
        x_t: another batch of images to concatenate with x (shape [B, 3, H, W])
        """
        
        t_embed = self.act(self.t_embed(t))
        
        input_x = torch.cat((x, x_t), dim=1)
        
        h0 = self.start_conv(input_x)
        h1 = self.conv1(h0, t_embed)
        h2 = self.conv2(h1, t_embed)
        h3 = self.conv3(h2, t_embed)
        out = self.conv4(h3, t_embed)
        
        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)
        
        out = self.final_conv(out)
        out = self.act(out)  
        
        out = out.view(out.size(0), -1)
        
        return out
    

#cloner174
