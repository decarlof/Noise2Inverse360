import torch
import torch.nn as nn
import torch.nn.functional as F

class unet_box_gn(torch.nn.Module):
    def __init__(self, in_ch, out_ch, groups):
        super().__init__()
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=groups, num_channels=out_ch),
            nn.LeakyReLU(.1, inplace=True),

            torch.nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=groups, num_channels=out_ch),
            nn.LeakyReLU(.1, inplace=True),

        )
    def forward(self, x):
        return self.double_conv(x)
    
class unet_bottleneck_gn(torch.nn.Module):
    def __init__(self, in_ch, out_ch, groups):
        super().__init__()
        self.bn_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=groups, num_channels=out_ch),
            nn.LeakyReLU(.1, inplace=True),
        )
    def forward(self, x):
        return self.bn_conv(x)
    
class unet_up(torch.nn.Module):
    def __init__(self, ch,):
        super().__init__()
        self.down_scale = torch.nn.Sequential(
                                torch.nn.Upsample(scale_factor=2, mode='nearest')
                            )

    def forward(self, x):
        return self.down_scale(x)
            
class unet_down(torch.nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.maxpool = torch.nn.Sequential(
            torch.nn.MaxPool2d(2), 
            )
        
    def forward(self, x):
        return self.maxpool(x)

class unet_ns_gn(torch.nn.Module):
    def __init__(self, start_filter_size, ich=1, och=1, channels_per_group=8):
        super().__init__()
        self.in_box= torch.nn.Sequential(
            torch.nn.Conv2d(ich, start_filter_size, kernel_size=1, padding=0), 
            nn.GroupNorm(num_groups=int((start_filter_size)/channels_per_group), num_channels=start_filter_size),
            nn.LeakyReLU(.1, inplace=True),
            )
        self.box1  = unet_box_gn(start_filter_size, start_filter_size*4, groups=int((start_filter_size*4)/channels_per_group))
        self.down1 = unet_down(start_filter_size*4)

        self.box2  = unet_box_gn(start_filter_size*4, start_filter_size*8, groups=int((start_filter_size*8)/channels_per_group))
        self.down2 = unet_down(start_filter_size*8)
        
        self.box3  = unet_box_gn(start_filter_size*8, start_filter_size*16, groups=int((start_filter_size*16)/channels_per_group))
        self.down3 = unet_down(start_filter_size*16)
        
        self.bottleneck = unet_bottleneck_gn(start_filter_size*16, start_filter_size*16, groups=int((start_filter_size*16)/channels_per_group))
        
        self.up1   = unet_up(start_filter_size*16)
        self.box4  = unet_box_gn(start_filter_size*16, start_filter_size*8, groups=int((start_filter_size*8)/channels_per_group))
        
        self.up2   = unet_up(start_filter_size*8)
        self.box5  = unet_box_gn(start_filter_size*8, start_filter_size*4, groups=int((start_filter_size*4)/channels_per_group))
        
        self.up3   = unet_up(start_filter_size*4)
        self.box6  = unet_box_gn(start_filter_size*4, start_filter_size*4, groups=int((start_filter_size*4)/channels_per_group))
        
        self.out_layer = torch.nn.Sequential(
            torch.nn.Conv2d(start_filter_size*4, start_filter_size*2, kernel_size=1, padding=0), 
            nn.GroupNorm(num_groups=int((start_filter_size*2)/channels_per_group), num_channels=start_filter_size*2),
            nn.LeakyReLU(.1, inplace=True),
            torch.nn.Conv2d(start_filter_size*2, och, kernel_size=1, padding=0), )
        
    def forward(self, x):
        output = self.in_box(x)

        output  = self.box1(output)
        output = self.down1(output)

        output  = self.box2(output)
        output = self.down2(output)
        
        output  = self.box3(output)
        output = self.down3(output)
        
        output = self.bottleneck(output)
        
        output     = self.up1(output)
        
        output    = self.box4(output)
        output     = self.up2(output)
        
        output    = self.box5(output)
        output     = self.up3(output)
        
        output    = self.box6(output)
        
        output      = self.out_layer(output)
        

        return output
    