# -*- coding: utf-8 -*-
"""
This Script contains the default and Spinal VGG code for EMNIST(Letters).
This code trains both NNs as two different models.
This code randomly changes the learning rate to get a good result.
@author: Dipu
"""


import torch
import torchvision
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
import os


from argparse import ArgumentParser
import copy
import time
import sys

#utils_path = '/ibex/user/ibraheao/path_project/utils'
utils_path = '/ibex/user/ibraheao/min_entropy_project/my_utils'
sys.path.insert(1, utils_path)
from my_utils import min_entropy_loss, mae_entropy_loss, norm_and_entropy_loss, mix_entropy_loss_2, cross_entropy_loss_base
from name_model import new_model_name

class LossModule(nn.Module):
    
    def __init__(self, learn_loss_weights = 1, loss_1_weight = 1.0, loss_2_weight = 1.0):
        super(LossModule, self).__init__()
        self.criterion_name = args.criterion
        self.loss_1_weight = nn.Parameter(torch.tensor([loss_1_weight])) if learn_loss_weights else torch.tensor([loss_1_weight]).to(device)
        self.loss_2_weight = nn.Parameter(torch.tensor([loss_2_weight])) if learn_loss_weights else torch.tensor([loss_2_weight]).to(device)

    def forward(self, out, targets, epoch):
        if self.criterion_name == 'cross_ent':
            criterion = nn.CrossEntropyLoss()
            loss = criterion(out, targets)

        elif self.criterion_name == 'cross_ent_b':
           
            loss = cross_entropy_loss_base(out, targets, eps=args.eps, base = args.base_1)

        elif self.criterion_name == 'norm_and_ent':
            loss = norm_and_entropy_loss(out, targets,  args.norm_power , self.loss_1_weight, self.loss_2_weight,  args.softmax_loss_weights,  args.tie_loss_weights, args.eps, args.base )
        
        elif self.criterion_name == 'mix_ent':
            loss = mix_entropy_loss(out, targets, self.loss_1_weight, self.loss_2_weight, args.softmax_loss_weights,  args.tie_loss_weights, args.eps, args.base)

        elif self.criterion_name == 'mix_ent_2':
            loss = mix_entropy_loss_2(out, targets, epoch, args.max_interval, loss_1_weight= self.loss_1_weight, loss_2_weight= self.loss_2_weight, loss_weights_style= args.loss_weights_style, eps= args.eps, base_1 = args.base_1,  base_2 = args.base_2 )

        elif self.criterion_name == 'min_ent':
            loss = min_entropy_loss(out, targets, epoch, args.max_interval, loss_1_weight= self.loss_1_weight, loss_2_weight= self.loss_2_weight, loss_weights_style= args.loss_weights_style, eps= args.eps, base_1 = args.base_1,  base_2 = args.base_2 )


        return out, loss





class VGG(nn.Module):  
    """
    Based on - https://github.com/kkweon/mnist-competition
    from: https://github.com/ranihorev/Kuzushiji_MNIST/blob/master/KujuMNIST.ipynb
    """
    def two_conv_pool(self, in_channels, f1, f2):
        s = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        for m in s.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return s
    
    def three_conv_pool(self,in_channels, f1, f2, f3):
        s = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.Conv2d(f2, f3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        for m in s.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return s
        
    
    def __init__(self, num_classes=62, learn_loss_weights = 1, loss_1_weight = 1.0, loss_2_weight = 1.0):
        super(VGG, self).__init__()
        self.l1 = self.two_conv_pool(1, 64, 64)
        self.l2 = self.two_conv_pool(64, 128, 128)
        self.l3 = self.three_conv_pool(128, 256, 256, 256)
        self.l4 = self.three_conv_pool(256, 256, 256, 256)
        
        self.classifier = nn.Sequential(
            nn.Dropout(p = args.dropout),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p = args.dropout),
            nn.Linear(512, num_classes),
        )

        self.loss_module = LossModule(learn_loss_weights = learn_loss_weights, loss_1_weight = loss_1_weight, loss_2_weight = loss_2_weight)      
        if learn_loss_weights and args.diff_lr == 1:
          self.params = nn.ModuleDict(   {  'base': nn.ModuleList( [self.l1, self.l2,self.l3,self.l4, self.classifier] ), 'loss_weights': nn.ModuleList( [self.loss_module] )  }   )


    
    def forward(self, x, targets, epoch):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        if args.criterion == 'cross_ent':
          x = F.log_softmax(x, dim = 1)
          x, loss = self.loss_module(x, targets, epoch)
        else:
          x, loss = self.loss_module(x, targets, epoch)
        return x, loss

    
    
Half_width =128
layer_width =128
    
class SpinalVGG(nn.Module):  
    """
    Based on - https://github.com/kkweon/mnist-competition
    from: https://github.com/ranihorev/Kuzushiji_MNIST/blob/master/KujuMNIST.ipynb
    """
    def two_conv_pool(self, in_channels, f1, f2):
        s = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        for m in s.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return s
    
    def three_conv_pool(self,in_channels, f1, f2, f3):
        s = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.Conv2d(f2, f3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        for m in s.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return s
        
    
    def __init__(self, num_classes=62, learn_loss_weights = 1, loss_1_weight = 1.0, loss_2_weight = 1.0):
        super(SpinalVGG, self).__init__()
        self.l1 = self.two_conv_pool(1, 64, 64)
        self.l2 = self.two_conv_pool(64, 128, 128)
        self.l3 = self.three_conv_pool(128, 256, 256, 256)
        self.l4 = self.three_conv_pool(256, 256, 256, 256)
        
        
        self.fc_spinal_layer1 = nn.Sequential(
            nn.Dropout(p = args.dropout), nn.Linear(Half_width, layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True),)
        self.fc_spinal_layer2 = nn.Sequential(
            nn.Dropout(p = args.dropout), nn.Linear(Half_width+layer_width, layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True),)
        self.fc_spinal_layer3 = nn.Sequential(
            nn.Dropout(p = args.dropout), nn.Linear(Half_width+layer_width, layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True),)
        self.fc_spinal_layer4 = nn.Sequential(
            nn.Dropout(p = args.dropout), nn.Linear(Half_width+layer_width, layer_width),
            nn.BatchNorm1d(layer_width), nn.ReLU(inplace=True),)
        self.fc_out = nn.Sequential(
            nn.Dropout(p = args.dropout), nn.Linear(layer_width*4, num_classes),)
        
        self.loss_module = LossModule(learn_loss_weights = learn_loss_weights, loss_1_weight = loss_1_weight, loss_2_weight = loss_2_weight)      
        if learn_loss_weights and args.diff_lr == 1:
          self.params = nn.ModuleDict(   {  'base': nn.ModuleList( [self.l1, self.l2,self.l3,self.l4, self.fc_spinal_layer1, self.fc_spinal_layer2, self.fc_spinal_layer3, self.fc_spinal_layer4, self.fc_out] ), 'loss_weights': nn.ModuleList( [self.loss_module] )  }   )

    
    def forward(self, x, targets, epoch):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = x.view(x.size(0), -1)
        
        x1 = self.fc_spinal_layer1(x[:, 0:Half_width])
        x2 = self.fc_spinal_layer2(torch.cat([ x[:,Half_width:2*Half_width], x1], dim=1))
        x3 = self.fc_spinal_layer3(torch.cat([ x[:,0:Half_width], x2], dim=1))
        x4 = self.fc_spinal_layer4(torch.cat([ x[:,Half_width:2*Half_width], x3], dim=1))
        
        x = torch.cat([x1, x2], dim=1)
        x = torch.cat([x, x3], dim=1)
        x = torch.cat([x, x4], dim=1)
        
        x = self.fc_out(x)

        if args.criterion == 'cross_ent':
          x = F.log_softmax(x, dim = 1)
          x, loss = self.loss_module(x, targets, epoch)
        else:
          x, loss = self.loss_module(x, targets, epoch)
        return x, loss




def test_fn(device, epoch, test_loader, model):

    model.eval()
    with torch.no_grad():

        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            
            labels = labels.to(device)
                        
            outputs, loss = model(images, labels, epoch)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return 100*correct/total




def main():
  
  device = 'cuda' 
  if args.model == 'vgg':
    model = VGG(learn_loss_weights = args.learn_loss_weights, loss_1_weight = args.loss_1_weight, loss_2_weight = args.loss_2_weight).to(device)
  elif args.model == 'spinal':
    model = SpinalVGG(learn_loss_weights = args.learn_loss_weights, loss_1_weight = args.loss_1_weight, loss_2_weight = args.loss_2_weight).to(device)



  checkpoint = torch.load(os.path.join(args.models_dir, args.input_check_name))
  model_state = checkpoint['model_state']
  model.load_state_dict(model_state)

  epoch = 0
  accuracy = test_fn(device, epoch, test_loader, model)
  print('Test Accuracy of Model: {} % '.format(accuracy), flush=True)
 



parser = ArgumentParser('SpinalNet')
parser.add_argument('--model', choices = ['vgg', 'spinal'], help='Loss function.')

parser.add_argument('--criterion', choices = ['cross_ent', 'cross_ent_b', 'min_ent', 'norm_and_ent', 'mix_ent', 'mix_ent_2'], help='Loss function.')
parser.add_argument('--eps', type=float, default=1e-6, help='Whether or not to use path info.')
parser.add_argument('--base_1', type=float, default=2.71828, help='Whether or not to use path info.')
parser.add_argument('--base_2', type=float, default=2.71828, help='Whether or not to use path info.')

parser.add_argument('--dropout', type=float, default=0.0, help='Whether or not to use path info.')

parser.add_argument('--diff_lr', type=int, default=1, help='Whether or not to use path info.')

parser.add_argument('--learn_loss_weights', type=int, default=1, help='Whether or not to use path info.')

parser.add_argument('--loss_1_weight', type=float, default=1.0, help='Whether or not to use path info.')
parser.add_argument('--loss_2_weight', type=float, default=1.0, help='Whether or not to use path info.')
parser.add_argument('--loss_weights_style', choices = ['free', 'square', 'square_tie', 'square_decay', 'square_decay_tie', 'square_wax', 'epoch_no_params', 'epoch_mult_params', 'epoch_add_params', 'epoch_add_mult',  'expo', 'softmax'], default='square_decay', help='Whether or not to use path info.')
parser.add_argument('--max_interval', type=int, default=None, help='Whether or not to use path info.')
parser.add_argument('--models_dir', type=str, default='/ibex/user/ibraheao/min_entropy_project/kabir/models_directory', help='Whether or not to use path info.')

parser.add_argument('--input_check_name', type=str, default=None, help='Whether or not to use path info.')

args = parser.parse_args()

print("ARGS: ", args)





data_dir = '/ibex/user/ibraheao/min_entropy_project/chopra/data/'

test_loader = torch.utils.data.DataLoader(torchvision.datasets.EMNIST(data_dir, split='letters', train=False, download=False,
              transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((0.1307,), (0.3081,))])),
              batch_size=1000, shuffle=True)


main()