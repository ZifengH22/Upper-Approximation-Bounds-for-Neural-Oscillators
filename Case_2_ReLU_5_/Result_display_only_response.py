import torch
import time
import numpy as np
from tqdm import tqdm 
from torch.utils.data import DataLoader  
from sklearn.model_selection import train_test_split 

import os 
import shutil 
import random 
import sys 
import contextlib 
                  
import scipy.io 
import matplotlib.pyplot as plt 
from Network_without_GRU_ResNet_simple_norm_RK2_simple_withoutbatchnorm import topDNN, myRK4GRUcell
import h5py 

class Args: 
    def __init__(self) -> None:  
        
        self.dt = 0.01                             
        self.SV_feature = 2                        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.layers = [5,10] 
        self.top_layers = [self.layers[-1],20,1]
        self.epochs = 5000
        
        self.data_path = r'data/'  
        self.modelsave_path = r'Results/'
        if not os.path.exists(self.modelsave_path):
            os.makedirs(self.modelsave_path)

def load_matdata(args):
    
    data_path_F = args.data_path + 'F_train.mat'
    data_path_X_dX_input_train = args.data_path + 'X_dX_input_train.mat'
    data_path_X_dX_output_train = args.data_path + 'X_dX_output_train.mat'
    data_path_E_X_output_train = args.data_path + 'E_X_output_train.mat'
    data_path_t_train = args.data_path + 't_train.mat'

    data_path_Acc = args.data_path + 'Acc_train.mat'
    data_path_X = args.data_path + 'X_train.mat'
    data_path_E_X = args.data_path + 'E_X_train.mat'
    
    data_path_coef_F = args.data_path + 'coef_F.mat'
    data_path_coef_X = args.data_path + 'coef_X_output.mat'
    data_path_coef_dX = args.data_path + 'coef_dX_output.mat'
    data_path_coef_ddX = args.data_path + 'coef_ddX_output.mat'
    data_path_coef_E_X = args.data_path + 'coef_E_X_output.mat'
    
    data_path_X_smallest = args.data_path + 'X_smallest.mat'
    
    ##training data    
    with h5py.File(data_path_F, 'r') as file:
        variable_name = list(file.keys())[0]
        F_input_train = file[variable_name][:]
    F_input_train = F_input_train[:, np.newaxis]
    F_input_train = np.transpose(F_input_train, axes = [2,0,1])
    
    with h5py.File(data_path_X_dX_input_train, 'r') as file:
        variable_name = list(file.keys())[0]
        X_dX_input_train = file[variable_name][:]
    X_dX_input_train = np.transpose(X_dX_input_train, axes = [2,1,0])

    with h5py.File(data_path_X_dX_output_train, 'r') as file:
        variable_name = list(file.keys())[0]
        X_dX_output_train = file[variable_name][:]
    X_dX_output_train = X_dX_output_train[:, np.newaxis]
    X_dX_output_train = np.transpose(X_dX_output_train, axes = [2,0,1])
    
    with h5py.File(data_path_E_X_output_train, 'r') as file:
        variable_name = list(file.keys())[0]
        E_X_output_train = file[variable_name][:]
    E_X_output_train = E_X_output_train[:, np.newaxis]
    E_X_output_train = np.transpose(E_X_output_train, axes = [2,0,1])
    
    with h5py.File(data_path_t_train, 'r') as file:
        variable_name = list(file.keys())[0]
        t_train = file[variable_name][:]
    t_train = t_train[:, np.newaxis]
    t_train = np.transpose(t_train, axes = [2,0,1])

    ##test data
    with h5py.File(data_path_Acc, 'r') as file:
        variable_name = list(file.keys())[0]
        Acc_test = file[variable_name][:]
    Acc_test = Acc_test[:, np.newaxis] 
    Acc_test = np.transpose(Acc_test, axes = [0,2,1])
    
    with h5py.File(data_path_X, 'r') as file:
        variable_name = list(file.keys())[0]
        X_test = file[variable_name][:]
    X_test = X_test[:, np.newaxis]
    X_test = np.transpose(X_test, axes = [0,2,1])

    with h5py.File(data_path_E_X, 'r') as file:
        variable_name = list(file.keys())[0]
        E_X_test = file[variable_name][:]
    E_X_test = E_X_test[:, np.newaxis]
    E_X_test = np.transpose(E_X_test, axes = [0,2,1])
    
    coef_F = scipy.io.loadmat(data_path_coef_F)
    coef_F = coef_F['coef_F'][0,0]

    coef_X = scipy.io.loadmat(data_path_coef_X)
    coef_X = coef_X['coef_X_output'][0,0]

    coef_dX = scipy.io.loadmat(data_path_coef_dX)
    coef_dX = coef_dX['coef_dX_output'][0,0]
    
    coef_ddX = scipy.io.loadmat(data_path_coef_ddX)
    coef_ddX = coef_ddX['coef_ddX_output'][0,0]

    coef_E_X = scipy.io.loadmat(data_path_coef_E_X)
    coef_E_X = coef_E_X['coef_E_X_output'][0,0]
    
    X_smallest = scipy.io.loadmat(data_path_X_smallest)
    X_smallest = X_smallest['X_smallest']

    return F_input_train, X_dX_input_train, X_dX_output_train, E_X_output_train, t_train, Acc_test, X_test, E_X_test,coef_F,coef_X,coef_dX,coef_ddX, coef_E_X, X_smallest

####################################
# test model
args = Args()
F_input_train, X_dX_input_train, X_dX_output_train, E_X_output_train, t_train, Acc_test, X_test, E_X_test,coef_F,coef_X,coef_dX,coef_ddX, coef_E_X, X_smallest  = load_matdata(args)

absxmax = np.max(np.abs(X_dX_output_train[:,:,0]))
F_input_train = torch.from_numpy(F_input_train)
X_dX_input_train = torch.from_numpy(X_dX_input_train)
X_dX_output_train = torch.from_numpy(X_dX_output_train)
t_train = torch.from_numpy(t_train)

# del Acc_test, X_test, E_X_test 
# test_exc = F_input_train
# test_x_xdot = X_dX_output_train
# test_initial = X_dX_input_train
# test_t = t_train
# num_sample_x_dx = test_x_xdot.shape[0]

Acc_test = torch.from_numpy(Acc_test)
test_exc = Acc_test
test_x_xdot = torch.from_numpy(X_test)
test_initial = torch.zeros(X_test.shape[0],1,X_test.shape[2]*2)
test_t = t_train
num_sample_x_dx = test_x_xdot.shape[0]
del Acc_test

###
num_input_layer_X_dX = torch.numel(X_dX_input_train[0,:,:])
num_input_layer_F = (torch.numel(F_input_train[0,:,0]) - torch.numel(X_dX_output_train[0,:,0]) + 1)*len(F_input_train[0,0,:])
num_input_layer = num_input_layer_X_dX + num_input_layer_F
args.layers.insert(0,num_input_layer_F + num_input_layer_X_dX*args.layers[-1])
args.top_layers[0] += (test_t.shape[2] + F_input_train.shape[2])

gru_step = torch.numel(X_dX_output_train[0,:,0])
step_delay_F = torch.numel(F_input_train[0,:,0]) - torch.numel(X_dX_output_train[0,:,0])
step_delay_X_dX = torch.numel(X_dX_input_train[0,:,0]) - 1

########
# load model
modelsave_path = args.modelsave_path
data_path = args.data_path

RK4GRUcell = myRK4GRUcell(args).to(args.device)
top_DNN = topDNN(args.top_layers, lastbias = True).to(args.device)

num_sample = scipy.io.loadmat(args.data_path + 'num_sample_select.mat')
str_layers = '_'.join(map(str, args.layers))
str_top_layers = '_'.join(map(str, args.top_layers))

# path_save_RK4GRUcell = modelsave_path + 'RK4GRUcell_best_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + str_top_layers + '.pth'
# path_save_topDNN = modelsave_path + 'topDNN_best_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + str_top_layers + '.pth'

path_save_RK4GRUcell = modelsave_path + 'RK4GRUcell_last_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + str_top_layers + '.pth'
path_save_topDNN = modelsave_path + 'topDNN_last_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + str_top_layers + '.pth'

# path_save_RK4GRUcell = modelsave_path + 'RK4GRUcell_smallest_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + str_top_layers + '.pth'
# path_save_topDNN = modelsave_path + 'topDNN_smallest_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + str_top_layers + '.pth'

print(path_save_RK4GRUcell)
print(path_save_topDNN)

RK4GRUcell.load_state_dict(torch.load(path_save_RK4GRUcell, map_location=torch.device('cpu')))
top_DNN.load_state_dict(torch.load(path_save_topDNN, map_location=torch.device('cpu')))

RK4GRUcell.eval()
top_DNN.eval()

#########calculate pred_response###########
num_sample_once = 200
iter_num = round(num_sample_x_dx/num_sample_once)
pred_state_final = torch.zeros_like(test_x_xdot).to(torch.float32)

for iter_id in range(iter_num):
    print(f'iter_id = {iter_id}')
    
    test_exc = test_exc.to(torch.float32).to(args.device)
    test_initial_temp = test_initial[iter_id*num_sample_once:(iter_id + 1)*num_sample_once,:,:].to(torch.float32).to(args.device)
    pred_state = torch.zeros_like(test_x_xdot[iter_id*num_sample_once:(iter_id + 1)*num_sample_once,:,:]).to(torch.float32).to(args.device) 
    SVi_delay_temp = torch.cat( (test_initial_temp[:,:,0::2].repeat(1, 1,args.layers[-1]), test_initial_temp[:,:,1::2].repeat(1, 1,args.layers[-1]) ),-1)        
    T_time = test_t[0:num_sample_once,:,:].to(torch.float32).to(args.device)

    svj = SVi_delay_temp
    top_DNN_input = torch.cat((svj[:,:,:svj.shape[2]//2],T_time[:,0:1,:],test_exc[iter_id*num_sample_once:(iter_id + 1)*num_sample_once,0:1,:]),-1)
    pred_state[:,0:1,:] = top_DNN(top_DNN_input)
    
    for i in tqdm(range(pred_state.shape[1] - step_delay_X_dX - 1), desc='Predict tracks'):
        exci_delay = test_exc[iter_id*num_sample_once:(iter_id + 1)*num_sample_once,i:(1 + i), :]
        excj = test_exc[iter_id*num_sample_once:(iter_id + 1)*num_sample_once,(i + 1):(i + 2), :]

        svj,_,_,_,_ = RK4GRUcell(SVi_delay_temp,step_delay_X_dX, step_delay_F,exci_delay,excj)
        top_DNN_input = torch.cat((svj[:,:,:svj.shape[2]//2],T_time[:,(i + 1):(i + 2),:],test_exc[iter_id*num_sample_once:(iter_id + 1)*num_sample_once,0:1,:]),-1)
        pred_state[:,i+1:i+2,:] = top_DNN(top_DNN_input)
        SVi_delay_temp = svj
   
    pred_state_final[iter_id*num_sample_once:(iter_id + 1)*num_sample_once,:,:] = pred_state.detach()

####saving data#####
X_pred = pred_state_final[:,:,0].numpy()
X_pred = np.transpose(pred_state_final[:,:,0].numpy(), axes = [1,0])
X_pred_dict = {'X_pred':X_pred  } 
scipy.io.savemat(args.data_path + 'X_pred.mat', X_pred_dict)  

##################plot#####################
test_x_xdot.shape
pred_state_final.shape

dt = args.dt
t = np.linspace(0, pred_state_final.shape[1]-1, pred_state_final.shape[1])*dt
index = 3456
plt.figure(1)
plt.plot(t,test_x_xdot[index,:,0],linestyle = '-',color = 'k')
plt.plot(t,pred_state_final[index,:,0],linestyle = '--',color = 'r')
plt.show()

plt.figure(2)
plt.plot(test_exc[index,:,0],linestyle = '-',color = 'k')
plt.show()

weight = torch.ones_like(test_x_xdot)
weight = torch.square(weight)
error_X_pred_all = (test_x_xdot - pred_state_final);
error_X_mean_all = torch.mean((error_X_pred_all*weight/coef_X)**2)
print(error_X_mean_all)

error_X_pred = (test_x_xdot[index,:,0] - pred_state_final[index,:,0]);
error_X_mean = torch.mean((error_X_pred*weight[index,:,:]/coef_X)**2)
print(error_X_mean)

plt.figure(3)
plt.plot(t,error_X_pred,linestyle = '--',color = 'r')
plt.show()

##########

path_train_loss = modelsave_path + 'train_epochs_loss_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + str_top_layers + '.npy'
path_val_loss = modelsave_path + 'val_epochs_loss_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + str_top_layers + '.npy'
print(path_val_loss)
print(path_train_loss)

train_loss = np.load(path_train_loss)
val_loss = np.load(path_val_loss)


plt.figure(4)
plt.plot(val_loss[-int(train_loss[:,1].size/2):,1],linestyle = '-',color = 'k')
plt.show()
print(val_loss[-10:,1])

plt.figure(5)
plt.plot(train_loss[-int(train_loss[:,1].size/2):,1],linestyle = '-',color = 'k')
plt.show()
print(train_loss[-10:,1])

########test on smallest SVj##############
path_save_train_indices = modelsave_path + 'train_indices_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + str_top_layers + '.npy'
path_save_val_indices = modelsave_path + 'val_indices_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + str_top_layers + '.npy'
train_indices = np.load(path_save_train_indices)
val_indices = np.load(path_save_val_indices)
X_smallest = X_smallest

dt = args.dt
t = np.linspace(0, pred_state_final.shape[1]-1, pred_state_final.shape[1])*dt

index = 110
train_indices[index]
plt.figure(6)
plt.plot(t,X_smallest[:,index],linestyle = '-',color = 'k')
plt.plot(t,test_x_xdot[train_indices[index],:,0],linestyle = '-.',color = 'b')
plt.plot(t,pred_state_final[train_indices[index],:,0],linestyle = '--',color = 'r')
plt.show()