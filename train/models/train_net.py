import torch
import torch.nn as nn #torch.nn是pytorch中自带的一个函数库，里面包含了神经网络中使用的一些常用函数
import torch.nn.functional as F
import torchvision.models as models #torchvision.models这个包中包含alexnet、densenet、inception、resnet、
# squeezenet、vgg等常用的网络结构，并且提供了预训练模型，可以通过简单调用来读取网络结构和预训练模型。

def conv_block(in_dim,out_dim):
    return nn.Sequential(nn.Conv2d(in_dim,in_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.Conv2d(in_dim,in_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.Conv2d(in_dim,out_dim,kernel_size=1,stride=1,padding=0),
                       nn.AvgPool2d(kernel_size=2,stride=2))
def deconv_block(in_dim,out_dim):
    return nn.Sequential(nn.Conv2d(in_dim,out_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.Conv2d(out_dim,out_dim,kernel_size=3,stride=1,padding=1),
                       nn.ELU(True),
                       nn.UpsamplingNearest2d(scale_factor=2))

def blockUNet1(in_c, out_c, name, transposed=False, bn=False, relu=True, dropout=False):
    block = nn.Sequential()
    if relu:
        block.add_module('%s.relu' % name, nn.ReLU(inplace=True))#inplace-可以选择就地执行操作默认为false
    else:
        block.add_module('%s.leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    if not transposed:
        block.add_module('%s.conv' % name, nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False))
    else:
        block.add_module('%s.tconv' % name, nn.ConvTranspose2d(in_c, out_c, 3, 1, 1, bias=False))
    if bn:
        block.add_module('%s.bn' % name, nn.BatchNorm2d(out_c))
    if dropout:
        block.add_module('%s.dropout' % name, nn.Dropout2d(0.5, inplace=True))
    return block

def blockUNet(in_c, out_c, name, transposed=False, bn=False, relu=True, dropout=False):
    block = nn.Sequential()
    if relu:
        block.add_module('%s-relu' % name, nn.ReLU(inplace=True))
    else:
        block.add_module('%s-leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    if not transposed:
        block.add_module('%s-conv' % name, nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False))
    else:
        block.add_module('%s-tconv' % name, nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False))#该函数是用来进行转置卷积的，
    if bn:
        block.add_module('%s-bn' % name, nn.BatchNorm2d(out_c))
    if dropout:
        block.add_module('%s-dropout' % name, nn.Dropout2d(0.5, inplace=True))
    return block


class D1(nn.Module):
    def __init__(self, nc, ndf, hidden_size):
        super(D1, self).__init__()

        self.conv1 = nn.Sequential(nn.Conv2d(nc,ndf,kernel_size=3,stride=1,padding=1), nn.ELU(True))
        self.conv2 = conv_block(ndf,ndf)
        self.conv3 = conv_block(ndf, ndf*2)
        self.conv4 = conv_block(ndf*2, ndf*3)
        self.encode = nn.Conv2d(ndf*3, hidden_size, kernel_size=1,stride=1,padding=0)
        self.decode = nn.Conv2d(hidden_size, ndf, kernel_size=1,stride=1,padding=0)
        self.deconv4 = deconv_block(ndf, ndf)
        self.deconv3 = deconv_block(ndf, ndf)
        self.deconv2 = deconv_block(ndf, ndf)
        self.deconv1 = nn.Sequential(nn.Conv2d(ndf,ndf,kernel_size=3,stride=1,padding=1),
                                 nn.ELU(True),
                                 nn.Conv2d(ndf,ndf,kernel_size=3,stride=1,padding=1),
                                 nn.ELU(True),
                                 nn.Conv2d(ndf, nc, kernel_size=3, stride=1, padding=1),
                                 nn.Tanh())

    def forward(self,x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)
        out4 = self.conv4(out3)
        out5 = self.encode(out4)
        dout5= self.decode(out5)
        dout4= self.deconv4(dout5)
        dout3= self.deconv3(dout4)
        dout2= self.deconv2(dout3)
        dout1= self.deconv1(dout2)
        return dout1

class D(nn.Module):
  def __init__(self, nc, nf):
    super(D, self).__init__()

    main = nn.Sequential()

    layer_idx = 1
    name = 'layer%d' % layer_idx
    main.add_module('%s-conv' % name, nn.Conv2d(nc, nf, 4, 2, 1, bias=False))

    layer_idx += 1
    name = 'layer%d' % layer_idx
    main.add_module(name, blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False))

    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module(name, blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False))

    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module('%s-leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    main.add_module('%s-conv' % name, nn.Conv2d(nf, nf*2, 4, 1, 1, bias=False))
    main.add_module('%s-bn' % name, nn.BatchNorm2d(nf*2))

    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module('%s-leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    main.add_module('%s-conv' % name, nn.Conv2d(nf, 1, 4, 1, 1, bias=False))
    main.add_module('%s-sigmoid' % name , nn.Sigmoid())

    self.main = main

  def forward(self, x):
    output = self.main(x)
    return output

class D_tran(nn.Module):
  def __init__(self, nc, nf):
    super(D_tran, self).__init__()

    main = nn.Sequential()

    layer_idx = 1
    name = 'layer%d' % layer_idx
    main.add_module('%s-conv' % name, nn.Conv2d(nc, nf, 4, 2, 1, bias=False))

    layer_idx += 1
    name = 'layer%d' % layer_idx
    main.add_module(name, blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False))

    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module(name, blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False))

    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module('%s-leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    main.add_module('%s-conv' % name, nn.Conv2d(nf, nf*2, 4, 1, 1, bias=False))
    main.add_module('%s-bn' % name, nn.BatchNorm2d(nf*2))

    layer_idx += 1
    name = 'layer%d' % layer_idx
    nf = nf * 2
    main.add_module('%s-leakyrelu' % name, nn.LeakyReLU(0.2, inplace=True))
    main.add_module('%s-conv' % name, nn.Conv2d(nf, 1, 4, 1, 1, bias=False))
    main.add_module('%s-sigmoid' % name , nn.Sigmoid())

    self.main = main

  def forward(self, x):
    output = self.main(x)
    return output



class G(nn.Module):
  def __init__(self, input_nc, output_nc, nf):
    super(G, self).__init__()

    layer_idx = 1
    name = 'layer%d' % layer_idx
    layer1 = nn.Sequential()
    layer1.add_module(name, nn.Conv2d(input_nc, nf, 4, 2, 1, bias=False))

    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer2 = blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False)

    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer3 = blockUNet(nf*2, nf*4, name, transposed=False, bn=True, relu=False, dropout=False)

    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer4 = blockUNet(nf*4, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)

    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer5 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)

    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer6 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)

    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer7 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)

    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer8 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)

    name = 'dlayer%d' % layer_idx
    d_inc = nf*8
    dlayer8 = blockUNet(d_inc, nf*8, name, transposed=True, bn=False, relu=True, dropout=True)

    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8*2
    dlayer7 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=True)

    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8*2
    dlayer6 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=True)

    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8*2
    dlayer5 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=False)

    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8*2
    dlayer4 = blockUNet(d_inc, nf*4, name, transposed=True, bn=True, relu=True, dropout=False)

    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*4*2
    dlayer3 = blockUNet(d_inc, nf*2, name, transposed=True, bn=True, relu=True, dropout=False)

    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*2*2
    dlayer2 = blockUNet(d_inc, nf, name, transposed=True, bn=True, relu=True, dropout=False)

    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    dlayer1 = nn.Sequential()
    d_inc = nf*2
    dlayer1.add_module('%s-relu' % name, nn.ReLU(inplace=True))
    dlayer1.add_module('%s-tconv' % name, nn.ConvTranspose2d(d_inc, 20, 4, 2, 1, bias=False))

    dlayerfinal = nn.Sequential()

    dlayerfinal.add_module('%s-conv' % name, nn.Conv2d(24, output_nc, 3, 1, 1, bias=False))
    dlayerfinal.add_module('%s-tanh' % name, nn.Tanh())

    self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)
    self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)
    self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)
    self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)

    self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)

    self.upsample = F.upsample_nearest

    self.layer1 = layer1
    self.layer2 = layer2
    self.layer3 = layer3
    self.layer4 = layer4
    self.layer5 = layer5
    self.layer6 = layer6
    self.layer7 = layer7
    self.layer8 = layer8
    self.dlayer8 = dlayer8
    self.dlayer7 = dlayer7
    self.dlayer6 = dlayer6
    self.dlayer5 = dlayer5
    self.dlayer4 = dlayer4
    self.dlayer3 = dlayer3
    self.dlayer2 = dlayer2
    self.dlayer1 = dlayer1
    self.dlayerfinal = dlayerfinal
    self.relu=nn.LeakyReLU(0.2, inplace=True)

  def forward(self, x):
    out1 = self.layer1(x)
    out2 = self.layer2(out1)
    out3 = self.layer3(out2)
    out4 = self.layer4(out3)
    out5 = self.layer5(out4)
    out6 = self.layer6(out5)
    out7 = self.layer7(out6)
    out8 = self.layer8(out7)
    dout8 = self.dlayer8(out8)
    dout8_out7 = torch.cat([dout8, out7], 1)
    dout7 = self.dlayer7(dout8_out7)
    dout7_out6 = torch.cat([dout7, out6], 1)
    dout6 = self.dlayer6(dout7_out6)
    dout6_out5 = torch.cat([dout6, out5], 1)
    dout5 = self.dlayer5(dout6_out5)
    dout5_out4 = torch.cat([dout5, out4], 1)
    dout4 = self.dlayer4(dout5_out4)
    dout4_out3 = torch.cat([dout4, out3], 1)
    dout3 = self.dlayer3(dout4_out3)
    dout3_out2 = torch.cat([dout3, out2], 1)
    dout2 = self.dlayer2(dout3_out2)
    dout2_out1 = torch.cat([dout2, out1], 1)
    dout1 = self.dlayer1(dout2_out1)

    shape_out = dout1.data.size()
    shape_out = shape_out[2:4]

    x101 = F.avg_pool2d(dout1, 16)
    x102 = F.avg_pool2d(dout1, 8)
    x103 = F.avg_pool2d(dout1, 4)
    x104 = F.avg_pool2d(dout1, 2)

    x1010 = self.upsample(self.relu(self.conv1010(x101)),size=shape_out)
    x1020 = self.upsample(self.relu(self.conv1020(x102)),size=shape_out)
    x1030 = self.upsample(self.relu(self.conv1030(x103)),size=shape_out)
    x1040 = self.upsample(self.relu(self.conv1040(x104)),size=shape_out)

    dehaze = torch.cat((x1010, x1020, x1030, x1040, dout1), 1)

    dout1 = self.dlayerfinal(dehaze)

    return dout1

class G2(nn.Module):
  def __init__(self, input_nc, output_nc, nf): # 3 3 8
    super(G2, self).__init__()

    layer_idx = 1
    name = 'layer%d' % layer_idx
    layer1 = nn.Sequential()
    layer1.add_module(name, nn.Conv2d(input_nc, nf, 4, 2, 1, bias=False))

    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer2 = blockUNet(nf, nf*2, name, transposed=False, bn=True, relu=False, dropout=False)

    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer3 = blockUNet(nf*2, nf*4, name, transposed=False, bn=True, relu=False, dropout=False)

    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer4 = blockUNet(nf*4, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)

    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer5 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)

    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer6 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)

    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer7 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)

    layer_idx += 1
    name = 'layer%d' % layer_idx
    layer8 = blockUNet(nf*8, nf*8, name, transposed=False, bn=True, relu=False, dropout=False)

    name = 'dlayer%d' % layer_idx
    d_inc = nf*8
    dlayer8 = blockUNet(d_inc, nf*8, name, transposed=True, bn=False, relu=True, dropout=True)

    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8*2
    dlayer7 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=True)

    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8*2
    dlayer6 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=True)

    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8*2
    dlayer5 = blockUNet(d_inc, nf*8, name, transposed=True, bn=True, relu=True, dropout=False)

    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*8*2
    dlayer4 = blockUNet(d_inc, nf*4, name, transposed=True, bn=True, relu=True, dropout=False)

    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*4*2
    dlayer3 = blockUNet(d_inc, nf*2, name, transposed=True, bn=True, relu=True, dropout=False)

    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    d_inc = nf*2*2
    dlayer2 = blockUNet(d_inc, nf, name, transposed=True, bn=True, relu=True, dropout=False)

    layer_idx -= 1
    name = 'dlayer%d' % layer_idx
    dlayer1 = nn.Sequential()
    d_inc = nf*2
    dlayer1.add_module('%s-relu' % name, nn.ReLU(inplace=True))
    dlayer1.add_module('%s-tconv' % name, nn.ConvTranspose2d(d_inc, output_nc, 4, 2, 1, bias=False))
    dlayer1.add_module('%s-tanh' % name, nn.LeakyReLU(0.2, inplace=True))

    self.layer1 = layer1
    self.layer2 = layer2
    self.layer3 = layer3
    self.layer4 = layer4
    self.layer5 = layer5
    self.layer6 = layer6
    self.layer7 = layer7
    self.layer8 = layer8
    self.dlayer8 = dlayer8
    self.dlayer7 = dlayer7
    self.dlayer6 = dlayer6
    self.dlayer5 = dlayer5
    self.dlayer4 = dlayer4
    self.dlayer3 = dlayer3
    self.dlayer2 = dlayer2
    self.dlayer1 = dlayer1

  def forward(self, x):
    out1 = self.layer1(x)
    out2 = self.layer2(out1)
    out3 = self.layer3(out2)
    out4 = self.layer4(out3)
    out5 = self.layer5(out4)
    out6 = self.layer6(out5)
    out7 = self.layer7(out6)
    out8 = self.layer8(out7)
    dout8 = self.dlayer8(out8)
    dout8_out7 = torch.cat([dout8, out7], 1)
    dout7 = self.dlayer7(dout8_out7)
    dout7_out6 = torch.cat([dout7, out6], 1)
    dout6 = self.dlayer6(dout7_out6)
    dout6_out5 = torch.cat([dout6, out5], 1)
    dout5 = self.dlayer5(dout6_out5)
    dout5_out4 = torch.cat([dout5, out4], 1)
    dout4 = self.dlayer4(dout5_out4)
    dout4_out3 = torch.cat([dout4, out3], 1)
    dout3 = self.dlayer3(dout4_out3)
    dout3_out2 = torch.cat([dout3, out2], 1)
    dout2 = self.dlayer2(dout3_out2)
    dout2_out1 = torch.cat([dout2, out1], 1)
    dout1 = self.dlayer1(dout2_out1)
    return dout1

class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(BottleneckBlock, self).__init__()
        inter_planes = out_planes * 4
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, inter_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(inter_planes)
        self.conv2 = nn.Conv2d(inter_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        out = self.conv2(self.relu(self.bn2(out)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return torch.cat([x, out], 1)

class TransitionBlock(nn.Module):
    def __init__(self, in_planes, out_planes, dropRate=0.0):
        super(TransitionBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.droprate = dropRate
    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, inplace=False, training=self.training)
        return F.upsample_nearest(out, scale_factor=2)

class Dense(nn.Module):
    def __init__(self):
        super(Dense, self).__init__()

        haze_class = models.densenet121(pretrained=True)

        self.conv0=haze_class.features.conv0
        self.norm0=haze_class.features.norm0
        self.relu0=haze_class.features.relu0
        self.pool0=haze_class.features.pool0

        self.dense_block1=haze_class.features.denseblock1
        self.trans_block1=haze_class.features.transition1

        self.dense_block2=haze_class.features.denseblock2
        self.trans_block2=haze_class.features.transition2

        self.dense_block3=haze_class.features.denseblock3
        self.trans_block3=haze_class.features.transition3

        self.dense_block4=BottleneckBlock(512,256)
        self.trans_block4=TransitionBlock(768,128)

        self.dense_block5=BottleneckBlock(384,256)
        self.trans_block5=TransitionBlock(640,128)

        self.dense_block6=BottleneckBlock(256,128)
        self.trans_block6=TransitionBlock(384,64)

        self.dense_block7=BottleneckBlock(64,64)
        self.trans_block7=TransitionBlock(128,32)

        self.dense_block8=BottleneckBlock(32,32)
        self.trans_block8=TransitionBlock(64,16)

        self.conv_refin=nn.Conv2d(19,20,3,1,1)
        self.tanh=nn.Tanh()

        self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)
        self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)
        self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)
        self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)

        self.refine3= nn.Conv2d(20+4, 3, kernel_size=3,stride=1,padding=1)

        self.upsample = F.upsample_nearest

        self.relu=nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):

        x0=self.pool0(self.relu0(self.norm0(self.conv0(x))))

        x1=self.dense_block1(x0)
        x1=self.trans_block1(x1)

        x2 = self.dense_block2(x1)

        x2=self.trans_block2(x2)

        x3 = self.dense_block3(x2)
        x3=self.trans_block3(x3)

        x4=self.trans_block4(self.dense_block4(x3))

        x42=torch.cat([x4,x2],1)
        x5=self.trans_block5(self.dense_block5(x42))

        x52=torch.cat([x5,x1],1)
        x6=self.trans_block6(self.dense_block6(x52))

        x7=self.trans_block7(self.dense_block7(x6))

        x8=self.trans_block8(self.dense_block8(x7))

        x8=torch.cat([x8,x],1)

        x9=self.relu(self.conv_refin(x8))

        shape_out = x9.data.size()

        shape_out = shape_out[2:4]

        x101 = F.avg_pool2d(x9, 32)

        x102 = F.avg_pool2d(x9, 16)
        x103 = F.avg_pool2d(x9, 8)
        x104 = F.avg_pool2d(x9, 4)

        x1010 = self.upsample(self.relu(self.conv1010(x101)), size=shape_out)
        x1020 = self.upsample(self.relu(self.conv1020(x102)), size=shape_out)
        x1030 = self.upsample(self.relu(self.conv1030(x103)), size=shape_out)
        x1040 = self.upsample(self.relu(self.conv1040(x104)), size=shape_out)

        dehaze = torch.cat((x1010, x1020, x1030, x1040, x9), 1)
        dehaze = self.tanh(self.refine3(dehaze))

        return dehaze

class ShareSepConv(nn.Module):
    def __init__(self, kernel_size):
        super(ShareSepConv, self).__init__()

        assert kernel_size % 2== 1
        self.padding = (kernel_size -1)//2
        weight_tensor = torch.zeros(1, 1, kernel_size, kernel_size)
        weight_tensor[0, 0, (kernel_size-1)//2, (kernel_size-1)//2] = 1
        self.weight = nn.Parameter(weight_tensor)
        self.kernel_size = kernel_size

    def forward(self, x):
        inc = x.size(1)
        expand_weight = self.weight.expand(inc, 1, self.kernel_size, self.kernel_size).contiguous()
        return  F.conv2d(x, expand_weight, None, 1, self.padding, 1, inc)

class SmoothDilated(nn.Module):
    def __init__(self, dilation=1, group=1):
        super(SmoothDilated, self).__init__()

        self.pre_conv1 = ShareSepConv(dilation*2-1)
        self.conv1 = nn.Conv2d(64, 64, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm1 = nn.InstanceNorm2d(64, affine=True)
        self.pre_conv2 = ShareSepConv(dilation*2-1)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, padding=dilation, dilation=dilation, groups=group, bias=False)
        self.norm2 = nn.InstanceNorm2d(64, affine=True)

    def forward(self, x):
        out1 = self.pre_conv1(x)
        out2 = self.conv1(out1)
        out3 = self.norm1(out2)

        return F.relu(x+out3)



class DCCL(nn.Module):
    def __init__(self):
        super(DCCL, self).__init__()

        # self.conv1 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        # self.conv2 = nn.Conv2d(64, 64, 3, 1, 3, dilation=3, bias=False)
        # self.conv3 = nn.Conv2d(64, 64, 3, 1, 5, dilation=5, bias=False)

        self.conv1 = SmoothDilated(dilation=1)
        self.conv2 = SmoothDilated(dilation=3)
        self.conv3 = SmoothDilated(dilation=5)

        self.conv = nn.Conv2d(192, 64, 1, 1, 0, bias=False)

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)

        out = torch.cat([out1, out2, out3], 1)

        final_out = self.conv(out)

        return final_out

class residual_block(nn.Module):
    def __init__(self):
        super(residual_block, self).__init__()

        self.DCCL_block1 = DCCL()
        self.BN1 = nn.BatchNorm2d(64)
        self.PReLU1 = nn.PReLU(num_parameters=1, init=0.25)
        #self.DCCL_block2 = DCCL()
        #self.BN2 = nn.BatchNorm2d(64)
        #self.PReLU2 = nn.PReLU(num_parameters=1, init=0.25)

    def forward(self, x):
        out1 = self.DCCL_block1(x)
        out2 = self.BN1(out1)
        out3 = self.PReLU1(out2)
        #out4 = self.DCCL_block2(out3)
        #out5 = self.BN2(out4)
        #out6 = self.PReLU2(out5)

        out = torch.add(out3, x)

        return out


class Detail_Net(nn.Module):
    def __init__(self):
        super(Detail_Net, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.PReLU1 = nn.PReLU(num_parameters=1, init=0.25)

        self.residual = residual_block()

        self.conv2 = nn.Conv2d(64, 64, 3, 1, 1,bias=False)
        self.BN1 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 3, 3, 1, 1, bias=False)


    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.PReLU1(out1)
        out = out2
        out3 = self.residual(out2)
        out4 = self.conv2(out3)
        out5 = self.BN1(out4)
        out6 = torch.add(out5, out)
        out7 = self.conv3(out6)

        return out7



class dehaze(nn.Module):
  def __init__(self, input_nc, output_nc, nf):
    super(dehaze, self).__init__()
    self.tran_est=G(input_nc=3,output_nc=3, nf=64)
    self.atp_est=G2(input_nc=3,output_nc=3, nf=8)

    self.tran_dense=Dense()
    self.relu=nn.LeakyReLU(0.2, inplace=True)

    self.tanh=nn.Tanh()

    self.refine1= nn.Conv2d(6, 20, kernel_size=3,stride=1,padding=1)
    self.refine2= nn.Conv2d(20, 20, kernel_size=3,stride=1,padding=1)
    self.threshold=nn.Threshold(0.1, 0.1)

    self.conv1010 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)
    self.conv1020 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)
    self.conv1030 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)
    self.conv1040 = nn.Conv2d(20, 1, kernel_size=1,stride=1,padding=0)

    self.detail = Detail_Net()

    self.refine3= nn.Conv2d(20+4+3, 3, kernel_size=3,stride=1,padding=1)

    self.upsample = F.upsample_nearest

    self.batch1= nn.BatchNorm2d(20)

  def forward(self, x):
    tran=self.tran_dense(x)
    atp= self.atp_est(x)
    zz= torch.abs((tran))+(10**-10)
    shape_out1 = atp.data.size()
    shape_out = shape_out1[2:4]
    atp = F.avg_pool2d(atp, shape_out1[2])
    atp = self.upsample(self.relu(atp),size=shape_out)
    dehaze= (x-atp)/zz+ atp
    dehaze2=dehaze
    dehaze=torch.cat([dehaze,x],1)
    dehaze=self.relu((self.refine1(dehaze)))
    dehaze=self.relu((self.refine2(dehaze)))
    shape_out = dehaze.data.size()

    shape_out = shape_out[2:4]

    x101 = F.avg_pool2d(dehaze, 32)
    x1010 = F.avg_pool2d(dehaze, 32)
    x102 = F.avg_pool2d(dehaze, 16)
    x1020 = F.avg_pool2d(dehaze, 16)
    x103 = F.avg_pool2d(dehaze, 8)
    x104 = F.avg_pool2d(dehaze, 4)

    x1010 = self.upsample(self.relu(self.conv1010(x101)),size=shape_out)
    x1020 = self.upsample(self.relu(self.conv1020(x102)),size=shape_out)
    x1030 = self.upsample(self.relu(self.conv1030(x103)),size=shape_out)
    x1040 = self.upsample(self.relu(self.conv1040(x104)),size=shape_out)

    y = self.detail(x)

    dehaze = torch.cat((x1010, x1020, x1030, x1040, dehaze, y), 1)
    dehaze= self.tanh(self.refine3(dehaze))

    return dehaze, tran, atp, dehaze2