import torch
import torch.nn as nn

# Creation of the architecture
class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):

        super().__init__()

        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels,out_channels, kernel_size = kernel_size, stride = 1, padding = kernel_size // 2, padding_mode = 'zeros', bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):

        return self.conv_block(x)

class Down(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size):

        super().__init__()

        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size)
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size = 1) 
        self.max_pooling = nn.MaxPool1d(2)

    def forward(self, x):

        skip = self.conv2(self.conv1(x)) + self.residual(x)

        return self.max_pooling(skip), skip

class Up(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, skip_pool):

        super().__init__()

        self.up = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2, output_padding=stride - 1) 
        self.skip_pool = nn.MaxPool1d(skip_pool) if skip_pool != 1 else nn.Identity()
        self.conv = ConvBlock(2 * out_channels, out_channels, kernel_size) 

    def forward(self, tensor1, tensor2):
         
        tensor1 = self.up(tensor1)
        tensor2 = self.skip_pool(tensor2)

        final_tensor = torch.cat([tensor1, tensor2], dim = 1)

        return self.conv(final_tensor)

class HookNet(nn.Module):

    def __init__(self, initial_filter = 24, n_layer = 13, downsampling_kernel = 15, upsampling_kernel = 5):
        
        super().__init__()

        n_input_channel = [initial_filter * (i + 1) for i in range(n_layer)]

        #DOWNSAMPLING

        self.downsampling = nn.ModuleList(

            [Down(1 if i == 0 else n_input_channel[i-1], n_input_channel[i], downsampling_kernel) for i in range(n_layer)]
                
        )

        #BOTTLENECK

        self.bottle_neck = ConvBlock(n_input_channel[-1],2 * n_input_channel[-1], downsampling_kernel)
        n_output_channel = list(reversed(n_input_channel))
        stride_list = [2 if i % 2 == 1 else 1 for i in range(n_layer)] 
        skip_pool_list = [2 ** (i // 2 + 1) for i in range(n_layer)] 

        #UPSAMPLING

        self.upsampling = nn.ModuleList(

            [Up(2 * n_input_channel[-1] if i == 0 else n_output_channel[i - 1] , n_output_channel[i], upsampling_kernel, stride_list[i], skip_pool_list[i]) for i in range(n_layer)]
                
        )

        #OUTPUT

        self.output_convoluted = nn.Conv1d(initial_filter, 1, kernel_size = 1)

    def forward(self, x):

        skips = []

        #ENCODER

        for down in self.downsampling:

            x, skip = down(x)
            skips.append(skip)

        #BOTTLENECK

        x = self.bottle_neck(x)

        #DECODER

        for up, skips in zip(self.upsampling , reversed(skips)):

            x = up(x, skips)

        #OUTPUT

        return torch.sigmoid(self.output_convoluted(x))
