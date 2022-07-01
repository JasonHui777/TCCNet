from cbam import ChannelAttentionModule
from cbam import SpatialAttentionModule
from cbam import CBAM
import torch

data = torch.normal(2, 3, size=(64, 1024, 21, 10,))
channel_attention_module = ChannelAttentionModule(1024)
channel_attention_module_data = channel_attention_module(data)
channel_attention_module_out = channel_attention_module_data * data

spatial_attention_module = SpatialAttentionModule()
spatial_attention_module_out = spatial_attention_module(channel_attention_module_out)
cbam = CBAM(1024)
print(channel_attention_module_out * spatial_attention_module_out)
