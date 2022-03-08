from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
from . import modelutils

def cbrp1d(in_fils, out_fils, rep, c_groups=1, n_groups=-1):
    ''' create EE-Block
    :param in_fils: the number of filters of input
    :param out_fils: the number of filters of output
    :param rep: the number of repeat of convolution layer
    :param c_groups: the number of groups in convolution layer (=1 is conventional convolution)
    :param n_groups: the number of groups in normalization (=-1 is BatchNorm, =1 is LayerNorm)
    :return: EE-Block
    '''
    layers = []
    i_f, o_f = in_fils, out_fils
    for _ in range(rep):
        layers.append(nn.Conv1d(i_f, o_f, kernel_size=3, padding=1, bias=False, groups=c_groups))
        if n_groups == -1:
            layers.append(nn.BatchNorm1d(o_f, affine=False))
        else:
            layers.append(nn.GroupNorm(n_groups, o_f, affine=False))
        layers.append(nn.ReLU(inplace=True))
        i_f = o_f
    layers.append(nn.MaxPool1d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)

class VGG_core(nn.Module):
    def __init__(self, in_channels=3, nb_fils=64, reps=[1,1,2,2,2], c_groups=1, n_groups=1):
        super(VGG_core, self).__init__()
        self.c_groups = c_groups
        self.n_groups = n_groups
        self.block1 = cbrp1d(in_channels, nb_fils, reps[0], c_groups=c_groups, n_groups=n_groups)
        self.block2 = cbrp1d(nb_fils, nb_fils * 2, reps[1], c_groups=c_groups, n_groups=n_groups)
        self.block3 = cbrp1d(nb_fils * 2, nb_fils * 4, reps[2], c_groups=c_groups, n_groups=n_groups)
        self.block4 = cbrp1d(nb_fils * 4, nb_fils * 8, reps[3], c_groups=c_groups, n_groups=n_groups)
        self.block5 = cbrp1d(nb_fils * 8, nb_fils * 8, reps[4], c_groups=c_groups, n_groups=n_groups)
        modelutils.initialize_weights(self)
        self.output_channels = nb_fils * 8

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = F.avg_pool1d(x, kernel_size=x.size()[2:])
        x = x.view(x.size(0), -1)
        return x

class VGG_head(nn.Module):
    def __init__(self, input_shape, num_classes):
        super(VGG_head, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_shape, num_classes, bias=False)
        )
        modelutils.initialize_weights(self)

    def forward(self, x):
        x = self.classifier(x)
        return x

class VGG(nn.Module):
    """ Backbone and classifier style vgg
    :param backbone: VGG's Encoder
    :param classifier: VGG's Classifier
    :param dropout: true or false
    :param lmd: random lamdas (lmd<0), no lambdas (lmd==0), and set same lambdas (lmd>0)
    :param repeat_input: the number of repeat of input
    """
    def __init__(self, backbone, classifier, dropout=False, lmd=-1, repeat_input=1):
        super(VGG, self).__init__()
        self.backbone = backbone
        self.groups = backbone.c_groups
        self.classifier = classifier
        self.dropout = None

        if lmd < 0:
            lambdas = np.random.normal(1, 0.3, size=self.groups)
            rmin = 0.2
            lambdas[lambdas < rmin] = rmin # floor
            lambdas = lambdas/np.sum(lambdas) # normalize
        elif lmd == 0:
            lambdas = np.repeat(1, self.groups)  # all lambdas are set to 1
        else:
            lambdas = np.repeat(lmd, self.groups)
            lambdas = lambdas/np.sum(lambdas) # normalize

        self.lambdas = np.repeat(lambdas, backbone.output_channels//len(lambdas))
        self.repeat_input = repeat_input
        if dropout:
            self.dropout = nn.Dropout(p=0.8)

    def forward(self, x):
        if self.repeat_input > 1:
            x = x.repeat(1, self.repeat_input, 1)
        x = self.backbone(x)
        if self.dropout is not None:
            x = self.dropout(x)
        if type(self.lambdas) == np.ndarray:
            self.lambdas = torch.Tensor(self.lambdas).to(x.device)
        x *= self.lambdas
        x = self.classifier(x)
        return x

def create_vgg8(num_classes, in_channels=3, nb_fils=64, c_groups=1, n_groups=1, dropout=False, repeat_input=1, lmd=1):
    if not (in_channels % c_groups == 0):
        print(f"in_channels ({in_channels}) could not divided by c_groups ({c_groups}), so the groups were set to 1.")
        c_groups = 1
    if not (in_channels % n_groups == 0):
        print(f"in_channels ({in_channels}) could not divided by n_groups ({n_groups}), so the groups were set to 1.")
        n_groups = 1
    backbone = VGG_core(in_channels, nb_fils, [1, 1, 2, 2, 2], c_groups=c_groups, n_groups=n_groups)
    classifier = VGG_head(backbone.output_channels, num_classes)
    m = VGG(backbone, classifier, dropout=dropout, lmd=lmd, repeat_input=repeat_input)
    return m
