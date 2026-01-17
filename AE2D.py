import torch
from torch import nn, optim, Tensor
import torch.nn.functional as F
from torch.autograd import Function
from typing import Optional, Any, Tuple
import numpy as np
import random
import math, copy

class BasicBlock_IBN(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, ibn=None, stride=1, downsample=None):
        super(BasicBlock_IBN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.IN = nn.InstanceNorm2d(planes, affine=True) if ibn == 'b' else None
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.IN is not None:
            out = self.IN(out)
        out = self.relu(out)

        return out

class AttentionBasicBlock_IBN(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, ibn=None, stride=1, downsample=None):
        super(AttentionBasicBlock_IBN, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.IN = nn.InstanceNorm2d(planes, affine=True) if ibn == 'b' else None
        self.downsample = downsample
        self.stride = stride
        self.attention = Self_Attention(inplanes, planes, stride, downsample)



    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        attout = self.attention(x)
        out += attout
        out += residual
        if self.IN is not None:
            out = self.IN(out)
        out = self.relu(out)

        return out

class ResNet_IBN(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 ibn_cfg=('b', 'b', None, None),
                 num_classes=1000):
        self.inplanes = 64
        super(ResNet_IBN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        if ibn_cfg[0] == 'b':
            self.bn1 = nn.InstanceNorm2d(64, affine=True)
        else:
            self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], ibn=ibn_cfg[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, ibn=ibn_cfg[1])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, ibn=ibn_cfg[2])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, ibn=ibn_cfg[3])
        # self.avgpool = nn.AvgPool2d(2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, ibn=None):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes,
                            None if ibn == 'b' else ibn,
                            stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes,
                                None if (ibn == 'b' and i < blocks-1) else ibn))

        return nn.Sequential(*layers)

    def forward(self, x):
        sfs = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # sfs.append(x)

        x = self.layer1(x)
        # sfs.append(x)
        x = self.layer2(x)
        # sfs.append(x)
        x = self.layer3(x)
        # sfs.append(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        sfs.append(x)

        return x, sfs


# ------------add self attention-----------
class Self_Attention(nn.Module):

    def __init__(self, in_dim, out_channel, stride=1, downsample=None, **kwargs):
        super(Self_Attention, self).__init__()
        self.chanel_in = in_dim
        # self.activation = activation
        ##  下面的query_conv，key_conv，value_conv即对应Wg,Wf,Wh
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 1, kernel_size=1)  # 即得到C^ X C 8
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 1, kernel_size=1)  # 即得到C^ X C 8
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)  # 即得到C X C
        self.gamma = nn.Parameter(torch.zeros(1))  # 这里即是计算最终输出的时候的伽马值，初始化为0
        # self.gamma = 1.
        self.sigmoid = nn.Sigmoid()

        self.softmax = nn.Softmax(dim=-1)
        self.downsample = downsample
        # self.conv2 = nn.Conv2d(in_channels=in_dim, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=out_channel,
                                             kernel_size=3, stride=stride, padding=1, bias=False),
                                   nn.BatchNorm2d(out_channel))

    def forward(self, x):
        m_batchsize, C, width, height = x.size()
        #  下面的proj_query，proj_key都是C^ X C X C X N= C^ X N
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N),permute即为转置
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check，进行点乘操作
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        # =====================
        # proj_query = self.query_conv(x).view(m_batchsize, -1, width * height)  # B X CX(N),permute即为转置
        # proj_key = self.key_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X C x (*W*H)
        # energy = torch.bmm(proj_query, proj_key) / math.sqrt(64)  # transpose check，进行点乘操作
        # attention = self.softmax(energy)  # BX (N) X (N)
        # proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N
        #
        # out = torch.bmm(attention, proj_value)
        # =====================
        out = out.view(m_batchsize, C, width, height)
        if self.downsample is not None:
            # out = nn.AvgPool2d(kernel_size=2, stride=1)(out)
            out = self.conv2(out)

        out = self.gamma * out
        # out = self.sigmoid(out)
        return out


class MultiHeadedAttention(nn.Module):
    def __init__(self,
                 num_heads: int,
                 d_model: int,
                 dropout: float = 0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        # Assume v_dim always equals k_dim
        self.k_dim = d_model // num_heads
        self.num_heads = num_heads
        self.proj_weights = clones(nn.Linear(d_model, d_model), 4)  # W^Q, W^K, W^V, W^O
        self.attention_score = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self,
                query: Tensor,
                key: Tensor,
                value: Tensor,
                mask: Optional[Tensor] = None):
        """
        Args:
            query: shape (batch_size, seq_len, d_model)
            key: shape (batch_size, seq_len, d_model)
            value: shape (batch_size, seq_len, d_model)
            mask: shape (batch_size, seq_len, seq_len). Since we assume all data use a same mask, so
                  here the shape also equals to (1, seq_len, seq_len)

        Return:
            out: shape (batch_size, seq_len, d_model). The output of a multihead attention layer
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
        batch_size = query.size(0)

        # 1) Apply W^Q, W^K, W^V to generate new query, key, value
        query, key, value \
            = [proj_weight(x).view(batch_size, -1, self.num_heads, self.k_dim).transpose(1, 2)
               for proj_weight, x in zip(self.proj_weights, [query, key, value])]  # -1 equals to seq_len

        # 2) Calculate attention score and the out
        out, self.attention_score = attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" output
        out = out.transpose(1, 2).contiguous() \
            .view(batch_size, -1, self.num_heads * self.k_dim)

        # 4) Apply W^O to get the final output
        out = self.proj_weights[-1](out)

        return out


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query: Tensor,
              key: Tensor,
              value: Tensor,
              mask: Optional[Tensor] = None,
              dropout: float = 0.1):
    """
    Define how to calculate attention score
    Args:
        query: shape (batch_size, num_heads, seq_len, k_dim)
        key: shape(batch_size, num_heads, seq_len, k_dim)
        value: shape(batch_size, num_heads, seq_len, v_dim)
        mask: shape (batch_size, num_heads, seq_len, seq_len). Since our assumption, here the shape is
              (1, 1, seq_len, seq_len)
    Return:
        out: shape (batch_size, v_dim). Output of an attention head.
        attention_score: shape (seq_len, seq_len).

    """
    k_dim = query.size(-1)

    # shape (seq_len ,seq_len)，row: token，col: that token's attention score
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(k_dim)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e10)

    attention_score = F.softmax(scores, dim=-1)

    if dropout is not None:
        attention_score = dropout(attention_score)

    out = torch.matmul(attention_score, value)

    return out, attention_score  # shape: (seq_len, v_dim), (seq_len, seq_lem)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out

class AttentionBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(AttentionBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample
        self.attention = Self_Attention(in_channel, out_channel, stride, downsample)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        attout = self.attention(x)

        out += attout
        out += identity
        out = self.relu(out)


        return out

class ResNet(nn.Module):
    def __init__(self,
                 block=BasicBlock,
                 blocks_num=[2, 2, 2, 2],
                 num_classes=2,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()

        # self.amp_norm = AmpNorm(input_shape=[3, 64, 64])

        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        # self.fc1 = nn.Linear(128*8*8, 128)
        # self.fc2 = nn.Linear(128*8*8, 128)

        # if self.include_top:
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        sfs = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)  #[128,8,8]
        x = self.layer3(x)

        x = self.layer4(x)   #[512,2,2]
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        sfs.append(x)
        return x, sfs




class Classifier(nn.Module):
    def __init__(self, num_classes=2):
        super(Classifier, self).__init__()
        
        self.fc = nn.Linear(256*2, num_classes)

    def forward(self, x):  #[512,2,2]
        
        x = self.fc(x)
        return x

class Bsaelineae(nn.Module): #============ AUC=

    def __init__(self):
        super().__init__()
        self.encoder1 = ResNet_IBN(BasicBlock_IBN, layers=[3, 4, 6, 3])
        self.cls1 = Classifier()


        pre_weight = torch.load('result/resnet34_ibn.pth')
        pre_dict = {k: v for k, v in pre_weight.items() if "fc" not in k}
        self.encoder1.load_state_dict(pre_dict, strict=False)

    def forward(self, x):
        f1, sfs1 = self.encoder1(x)
        clas1 = self.cls1(f1)

        return clas1, sfs1

class Bsaelineresnet(nn.Module): #============ AUC=

    def __init__(self):
        super().__init__()
        self.encoder1 = ResNet(BasicBlock, blocks_num=[3, 4, 6, 3])
        self.cls1 = Classifier()


        pre_weight = torch.load('result/resnet34-333f7ec4.pth')
        pre_dict = {k: v for k, v in pre_weight.items() if "fc" not in k}
        self.encoder1.load_state_dict(pre_dict, strict=False)

    def forward(self, x):
        f1, sfs1 = self.encoder1(x)
        clas1 = self.cls1(f1)

        return clas1, sfs1

class Bsaelineattention(nn.Module): #============ AUC=

    def __init__(self):
        super().__init__()
        self.encoder1 = ResNet(AttentionBasicBlock, blocks_num=[3, 4, 6, 3])
        self.cls1 = Classifier()


        pre_weight = torch.load('result/resnet34-333f7ec4.pth')
        pre_dict = {k: v for k, v in pre_weight.items() if "fc" not in k}
        self.encoder1.load_state_dict(pre_dict, strict=False)

    def forward(self, x):
        f1, sfs1 = self.encoder1(x)
        clas1 = self.cls1(f1)

        return clas1, sfs1

# 解耦带有域标签
class Disentae(nn.Module): #============ AUC=

    def __init__(self, mixstyle_layers=[]):
        super().__init__()
        self.encoder1 = ResNet_IBN(BasicBlock_IBN, layers=[3, 4, 6, 3])
        
        self.cls1 = Classifier()
       

        pre_weight = torch.load('result/resnet34_ibn.pth')
        pre_dict = {k: v for k, v in pre_weight.items() if "fc" not in k}
        self.encoder1.load_state_dict(pre_dict, strict=False)
      


    def forward(self, x):
        
        f1, sfs1 = self.encoder1(x)
        
        clas1 = self.cls1(f1)
        #

        f11 = torch.flip(f1, dims=(0,))
       
        

        return clas1, sfs1, clas1, sfs1, clas1, f1, f11


class ResNet34Dec(nn.Module):
    def __init__(self, hidden_dims=[32, 64, 128, 256, 512]):
        super(ResNet34Dec, self).__init__()
        self.decoder_input = nn.Linear(512*2, 512 * 4)
        self.decoder = self.builddecoder(hidden_dims)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Sigmoid())


    def forward(self, x):
        x = self.decoder_input(x)
        x = x.view(-1, 512, 2, 2)
        x = self.decoder(x)
        x = self.final_layer(x)
        return x

    def builddecoder(self, hidden_dims):
        modules = []
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()))
        decoder = nn.Sequential(*modules)
        return decoder

class ResNet34Dec224(nn.Module):
    def __init__(self, hidden_dims=[32, 64, 128, 256, 512]):
        super(ResNet34Dec224, self).__init__()
        self.decoder_input = nn.Linear(512*2, 512 * 2*2)
        self.decoder = self.builddecoder(hidden_dims)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Sigmoid())


    def forward(self, x):
        x = self.decoder_input(x)
        x = x.view(-1, 512, 2, 2)
        x = self.decoder(x)
        x = self.final_layer(x)
        return x

    def builddecoder(self, hidden_dims):
        modules = []
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()))
        decoder = nn.Sequential(*modules)
        return decoder

class ResNet34Dec2242(nn.Module):
    def __init__(self, hidden_dims=[32, 64, 128, 256, 512]):
        super(ResNet34Dec2242, self).__init__()
        self.decoder_input = nn.Linear(512*1, 512 * 2*2)
        self.decoder = self.builddecoder(hidden_dims)
        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dims[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Sigmoid())


    def forward(self, x):
        x = self.decoder_input(x)
        x = x.view(-1, 512, 2, 2)
        x = self.decoder(x)
        x = self.final_layer(x)
        return x

    def builddecoder(self, hidden_dims):
        modules = []
        hidden_dims.reverse()
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()))
        decoder = nn.Sequential(*modules)
        return decoder


class DisentReconae224(nn.Module): 

    def __init__(self, mixstyle_layers=[]):
        super().__init__()
        self.encoder1 = ResNet_IBN(BasicBlock_IBN, layers=[3, 4, 6, 3])
        self.encoder2 = ResNet_IBN(BasicBlock_IBN, layers=[3, 4, 6, 3])
        self.decoder = ResNet34Dec224(hidden_dims=[32, 64, 128, 256, 512])
        
        self.cls1 = Classifier()
        self.cls2 = Classifier()
        self.classifier_task = Classifier()




    def forward(self, x):

        f1, sfs1 = self.encoder1(x)
        f2, sfs2 = self.encoder2(x)

        clas1 = self.cls1(f1)
        clas2 = self.cls2(f2)

        f11 = torch.flip(f1, dims=(0,))

        f3 = torch.cat((f1, f2))
        clast = self.classifier_task(f3)

        rz = torch.cat([f1, f2], 1)
        rec = self.decoder(rz)
        rec = x

        return clas1, sfs1, clas2, sfs2, clast, f1, f11, rec

