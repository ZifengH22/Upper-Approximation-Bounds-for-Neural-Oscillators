import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        # self.bn1 = nn.BatchNorm1d(output_dim)

    def forward(self, x):
        out = self.fc1(x)
        # out = F.relu(self.bn1(self.fc1(x)))
        return out


class myDNN(nn.Module):
    def __init__(self, layers,lastbias = False): 
        super(myDNN, self).__init__()
        self.depth = len(layers) - 1

        # set up layer order dict
        self.activation = torch.nn.ReLU
        # self.activation = torch.nn.PReLU

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(('mlp_layer_%d' % i, nn.Linear(layers[i], layers[i + 1])))
            layer_list.append(('activation_%d' % i, self.activation()))        # layers[i + 1]   
        layer_list.append(('output_layer', nn.Linear(layers[-2], layers[-1], bias=lastbias)))
        # layer_list.append(('residual_block', ResidualBlock(layers[0], layers[-1])))
        ####
        layerDict = OrderedDict(layer_list)
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = x  # Initialize out with the input x
        for name, layer in self.layers.named_children():
            if 'mlp_layer_' in name:
                mlp_out = layer(out)  # Process input through MLP layer
            elif 'activation_' in name:
                out = layer(mlp_out)  # Process MLP output through activation          
            elif 'output_layer' in name:
                out = layer(out)  # Final output layer processing
            # elif 'residual_block' in name:
            #     residual_out = layer(x)  # Process input through residual block
            #     out = out + residual_out  # Combine MLP+activation output with residual block output
        return out


# RK4GRUcell
class myRK4GRUcell(nn.Module):
    def __init__(self, args):
        super(myRK4GRUcell, self).__init__()
        self.layers = args.layers
        self.dt = args.dt
        self.DNN = myDNN(self.layers, lastbias = True)
        
    def forward(self, SVi_delay_temp, step_delay_X_dX, step_delay_F, exci_delay = None, excj = None):
        SVi_r = SVi_delay_temp[:,-1,:].unsqueeze(1) 

        #### K1 ####
        SVK_INPUT1 = SVi_r
        NET_INPUT_X_dX1 = SVK_INPUT1
        NET_INPUT_F1 = exci_delay                          
        NET_INPUT1 = torch.cat((NET_INPUT_X_dX1,NET_INPUT_F1),-1)
        K1 = torch.cat((SVK_INPUT1[:,:,SVK_INPUT1.shape[2]//2:], self.DNN( NET_INPUT1 )) ,-1)
       
        #### K2 #### 
        SVK_INPUT2 = SVi_r + (self.dt)*K1
        NET_INPUT_X_dX2 = SVK_INPUT2
        NET_INPUT_F2 = excj                                   
        NET_INPUT2 = torch.cat((NET_INPUT_X_dX2,NET_INPUT_F2),-1)
        K2 = torch.cat((SVK_INPUT2[:,:,SVK_INPUT2.shape[2]//2:], self.DNN( NET_INPUT2 )) ,-1)
           
        ######## final response ########
        SVj_z = SVi_r + (self.dt/2)*(K1 + K2)
        return SVj_z,K1,K2,K1,K2


class topDNN(nn.Module):
    def __init__(self, layers,lastbias = False):
        super(topDNN, self).__init__()
        self.depth = len(layers) - 1

        # set up layer order dict
        self.activation = torch.nn.ReLU
        # self.activation = torch.nn.PReLU

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(('mlp_layer_%d' % i, nn.Linear(layers[i], layers[i + 1])))
            layer_list.append(('activation_%d' % i, self.activation()))        # layers[i + 1]   
        layer_list.append(('output_layer', nn.Linear(layers[-2], layers[-1], bias=lastbias)))
        # layer_list.append(('residual_block', ResidualBlock(layers[0], layers[-1])))
        ####
        layerDict = OrderedDict(layer_list)
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = x  # Initialize out with the input x
        for name, layer in self.layers.named_children():
            if 'mlp_layer_' in name:
                mlp_out = layer(out)  # Process input through MLP layer
            elif 'activation_' in name:
                out = layer(mlp_out)  # Process MLP output through activation          
            elif 'output_layer' in name:
                out = layer(out)  # Final output layer processing
            # elif 'residual_block' in name:
            #     residual_out = layer(x)  # Process input through residual block
            #     out = out + residual_out  # Combine MLP+activation output with residual block output
        return out







