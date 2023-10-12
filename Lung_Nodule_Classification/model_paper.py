from torch import nn as nn

class Base_NN_Model(nn.Module):
    def __init__(self, in_channels=1, conv_channels=32):
        super().__init__()

        self.block1 = ConvBlock(in_channels, conv_channels)
        self.block2 = ConvBlock(conv_channels, conv_channels * 2)
        self.block3 = ConvBlock(conv_channels * 2, conv_channels * 4)

        self.FC1 = nn.Linear((conv_channels * 4)*4*4*4, 1024)
        self.FC1_PRelu = nn.PReLU()
        self.dropout = nn.Dropout3d(p=0.5, inplace=True)
        self.FC2 = nn.Linear(1024,2)

#         self._init_weights()

#     # see also https://github.com/pytorch/pytorch/issues/18182
#     def _init_weights(self):
#         for m in self.modules():
#             if type(m) in {
#                 nn.Linear,
#                 nn.Conv3d,
#                 nn.Conv2d,
#                 nn.ConvTranspose2d,
#                 nn.ConvTranspose3d,
#             }:
#                 nn.init.kaiming_normal_(
#                     m.weight.data, a=0, mode='fan_out', nonlinearity='relu',
#                 )
#                 if m.bias is not None:
#                     fan_in, fan_out = \
#                         nn.init._calculate_fan_in_and_fan_out(m.weight.data)
#                     bound = 1 / math.sqrt(fan_out)
#                     nn.init.normal_(m.bias, -bound, bound)



    def forward(self, input_batch):

        block_out = self.block1(input_batch)
        block_out = self.block2(block_out)
        block_out = self.block3(block_out)
    

        conv_flat = block_out.view(block_out.size(0),-1,)
        linear_output1 = self.FC1(conv_flat)
        linear_output1 = self.FC1_PRelu(linear_output1)
        linear_output1 = self.dropout(linear_output1)
        linear_output2 = self.FC2(linear_output1)

        return linear_output2
        
        
class ConvBlock(nn.Module):
    def __init__(self, in_channels, conv_channels):
        super().__init__()

        self.conv = nn.Conv3d(in_channels, conv_channels, kernel_size=5, stride=1, padding=2, bias=True,)
        self.conv_PRelu = nn.PReLU()
        self.conv_batchnorm = nn.BatchNorm3d(num_features=conv_channels)
        self.maxpool = nn.MaxPool3d(2, stride=2)

    def forward(self, input_batch):
        block_out = self.conv(input_batch)
        block_out = self.conv_PRelu(block_out)
        block_out = self.conv_batchnorm(block_out)
  
        return self.maxpool(block_out)