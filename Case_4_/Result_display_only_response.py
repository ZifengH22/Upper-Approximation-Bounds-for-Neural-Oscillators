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
        
        self.dt = 0.005                             
        self.SV_feature = 2                        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.layer_width = 2
        self.layers = [self.layer_width,20] 
        self.top_layers = [self.layers[-1],self.layer_width,1]
        self.epochs = 10000
        
        self.data_path = r'data/'  
        self.modelsave_path = r'Results/'
        if not os.path.exists(self.modelsave_path):
            os.makedirs(self.modelsave_path)

def load_matdata(args):
    data_path_Acc = args.data_path + 'Acc.mat'
    data_path_Acc_train = args.data_path + 'Acc_train.mat'
    data_path_X_dX_input_test = args.data_path + 'X_dX_input_test.mat'
    data_path_X_dX_output_test = args.data_path + 'X200_response.mat'
    data_path_t_train = args.data_path + 't_train.mat'

    with h5py.File(data_path_Acc, 'r') as file:
        variable_name = list(file.keys())[0]
        Acc_input = file[variable_name][:]
    Acc_input = Acc_input[:, np.newaxis]
    Acc_input = np.transpose(Acc_input, axes = [0,2,1])

    with h5py.File(data_path_Acc_train, 'r') as file:
        variable_name = list(file.keys())[0]
        Acc_train = file[variable_name][:]
    Acc_train = Acc_train[:, np.newaxis]
    Acc_train = np.transpose(Acc_train, axes = [2,0,1])

    with h5py.File(data_path_X_dX_input_test, 'r') as file:
        variable_name = list(file.keys())[0]
        X_dX_input_test = file[variable_name][:]
    # X_dX_input_test = X_dX_input_test[:, np.newaxis] 
    X_dX_input_test= np.transpose(X_dX_input_test, axes = [2,1,0])

    with h5py.File(data_path_X_dX_output_test, 'r') as file:
        variable_name = list(file.keys())[0]
        X_dX_output_test = file[variable_name][:]
    X_dX_output_test = X_dX_output_test[:, np.newaxis]
    X_dX_output_test = np.transpose(X_dX_output_test, axes = [2,0,1])

    with h5py.File(data_path_t_train, 'r') as file:
        variable_name = list(file.keys())[0]
        t_train = file[variable_name][:]
    t_train = t_train[:, np.newaxis]
    t_train = np.transpose(t_train, axes = [2,0,1])

    return Acc_input, Acc_train, X_dX_input_test,X_dX_output_test, t_train

####################################
# test model
args = Args()
Acc_test, Acc_train, X_dX_input_test, X_dX_output_test, t_test  = load_matdata(args)


Acc_train = torch.from_numpy(Acc_train)
test_exc = torch.from_numpy(Acc_test)
test_initial = torch.from_numpy(X_dX_input_test)
test_X = torch.from_numpy(X_dX_output_test)
test_t = torch.from_numpy(t_test)
num_sample_x_dx = test_initial.shape[0]
del Acc_test, X_dX_output_test

###
num_input_layer_Acc = len(test_exc[0,0,:])
num_input_layer_X_dX = 2*num_input_layer_Acc
args.layers.insert(0,num_input_layer_Acc + num_input_layer_X_dX*args.layers[-1])
args.top_layers[0] += (test_t.shape[2] + test_exc.shape[2])
gru_step = torch.numel(test_exc[0,:,0])

########
# load model
modelsave_path = args.modelsave_path
data_path = args.data_path

RK4GRUcell = myRK4GRUcell(args).to(args.device)
top_DNN = topDNN(args.top_layers, lastbias = True).to(args.device)

num_sample = scipy.io.loadmat(args.data_path + 'num_sample_select.mat')
str_layers = '_'.join(map(str, args.layers))

# path_save_RK4GRUcell = modelsave_path + 'RK4GRUcell_best_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + '.pth'
# path_save_top_DNN = modelsave_path + 'top_DNN_model_best_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + '.pth'

path_save_RK4GRUcell = modelsave_path + 'RK4GRUcell_last_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + '.pth'
path_save_top_DNN = modelsave_path + 'top_DNN_model_last_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + '.pth'

# path_save_RK4GRUcell = modelsave_path + 'RK4GRUcell_smallest_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + '.pth'
# path_save_top_DNN = modelsave_path + 'top_DNN_model_smallest_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + '.pth'

print(path_save_RK4GRUcell)
print(path_save_top_DNN)

RK4GRUcell.load_state_dict(torch.load(path_save_RK4GRUcell, map_location=torch.device('cpu'),weights_only=True))
top_DNN.load_state_dict(torch.load(path_save_top_DNN, map_location=torch.device('cpu'),weights_only=True))

RK4GRUcell.eval()
top_DNN.eval()

#########calculate pred_response###########
num_sample_once = 200
iter_num = round(num_sample_x_dx/num_sample_once)
pred_state_final = torch.zeros_like(test_X).to(torch.float32)
test_exc = test_exc.to(torch.float32).to(args.device)

for iter_id in range(iter_num):
    print(f'iter_id = {iter_id}')    
    
    test_exc_temp = test_exc[iter_id*num_sample_once:(iter_id + 1)*num_sample_once,:,:]
    test_initial_temp = test_initial[iter_id*num_sample_once:(iter_id + 1)*num_sample_once,:,:].to(torch.float32).to(args.device)
    # pred_state = torch.zeros((test_initial_temp.shape[0],test_X.shape[1],args.layers[-1])).to(torch.float32).to(args.device)
    pred_state = torch.zeros_like(test_X[iter_id*num_sample_once:(iter_id + 1)*num_sample_once,:,:]).to(torch.float32).to(args.device) 
    SVi_delay_temp = torch.cat( (test_initial_temp[:,:,0::2].repeat(1, 1,args.layers[-1]), test_initial_temp[:,:,1::2].repeat(1, 1,args.layers[-1]) ),-1)        
    T_time = test_t[0:num_sample_once,:,:].to(torch.float32).to(args.device)

    svj = SVi_delay_temp
    top_DNN_input = torch.cat((svj[:,:,:svj.shape[2]//2],T_time[:,0:1,:],test_exc[iter_id*num_sample_once:(iter_id + 1)*num_sample_once,0:1,:]),-1)
    pred_state[:,0:1,:] = top_DNN(top_DNN_input)
    
    for i in tqdm(range(gru_step - 1), desc='Predict tracks'):
        exci_delay = test_exc[iter_id*num_sample_once:(iter_id + 1)*num_sample_once,i:(1 + i), :]
        excj = test_exc[iter_id*num_sample_once:(iter_id + 1)*num_sample_once,(i + 1):(i + 2), :]
        svj,_,_,_,_ = RK4GRUcell(SVi_delay_temp,0.0, 0.0,exci_delay,excj)
        top_DNN_input = torch.cat((svj[:,:,:svj.shape[2]//2],T_time[:,(i + 1):(i + 2),:],test_exc[iter_id*num_sample_once:(iter_id + 1)*num_sample_once,0:1,:]),-1)
        pred_state[:,i+1:i+2,:] = top_DNN(top_DNN_input)
        SVi_delay_temp = svj
    
    pred_state_final[iter_id*num_sample_once:(iter_id + 1)*num_sample_once,:,:] = pred_state.detach()
  

####saving data#####
X_pred = pred_state_final[:,:,0].numpy()
X_pred = np.transpose(pred_state_final[:,:,0].numpy(), axes = [0,1])
X_pred_dict = {'X_pred_'+str(args.layer_width):X_pred  } 
scipy.io.savemat(args.data_path + 'X_pred_'+str(args.layer_width)+'.mat', X_pred_dict)  


##################plot#####################
test_exc.shape
pred_state_final.shape

dt = args.dt
t = np.linspace(0, pred_state_final.shape[1]-1, pred_state_final.shape[1])*dt
index = 3456
plt.figure(1)
plt.plot(t,test_X[index,:,0],linestyle = '-',color = 'k')
plt.plot(t,pred_state_final[index,:,0],linestyle = '--',color = 'r')
plt.show()

plt.figure(2)
plt.plot(test_exc[index,:,0],linestyle = '-',color = 'k')
plt.show()

####
error_mat = torch.abs(test_X - pred_state_final)
loss = torch.mean(error_mat**2)
loss_max = torch.max(error_mat)

print("loss is ",format(loss))
print("loss_max is ",format(loss_max))


##########

