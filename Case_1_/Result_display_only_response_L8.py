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
from Network_without_GRU_ResNet_simple_norm_RK2_simple import topDNN, myRK4GRUcell
import h5py 

class Args: 
    def __init__(self) -> None:  
        
        self.dt = 0.01                             
        self.SV_feature = 2                       
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.layers = [10,5]   #M_Gamma*p
        self.top_layers = [self.layers[-1],1]
        self.hidden_number = 5
        self.lamda = 1
        self.Lp = 8
        self.loss_opt = 0
        self.epochs = 10000
        
        ### file path
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

F_input_train = torch.from_numpy(F_input_train)
X_dX_input_train = torch.from_numpy(X_dX_input_train)
X_dX_output_train = torch.from_numpy(X_dX_output_train)
E_X_output_train = torch.from_numpy(E_X_output_train)
t_train = torch.from_numpy(t_train)

# del Acc_test, X_test, E_X_test 
# test_exc = F_input_train
# test_x_xdot = E_X_output_train
# test_initial = X_dX_input_train
# test_t = t_train
# num_sample_x_dx = test_x_xdot.shape[0]

Acc_test = torch.from_numpy(Acc_test)
test_exc = Acc_test
test_x_xdot = torch.from_numpy(E_X_test)
test_initial = torch.zeros(E_X_test.shape[0],1,E_X_test.shape[2]*2)
test_t = t_train
num_sample_x_dx = test_x_xdot.shape[0]
del Acc_test

###
num_input_layer_X_dX = torch.numel(X_dX_input_train[0,:,:])
num_input_layer_F = len(F_input_train[0,0,:])
num_input_layer = num_input_layer_X_dX + num_input_layer_F
hidden_number = args.hidden_number

args.layers.insert(0,num_input_layer_F + num_input_layer_X_dX*args.layers[-1])
args.top_layers[0] += (test_t.shape[2] + num_input_layer_F)
q = args.top_layers[-1]
for i in range(args.hidden_number): 
    args.top_layers.insert(i+1,(args.layers[-1]+num_input_layer_F+4)*q)
del q

gru_step = torch.numel(E_X_output_train[0,:,0])
step_delay_F = torch.numel(F_input_train[0,:,0]) - torch.numel(X_dX_output_train[0,:,0])
step_delay_X_dX = torch.numel(X_dX_input_train[0,:,0]) - 1

########
# load model
modelsave_path = args.modelsave_path
data_path = args.data_path

RK4GRUcell = myRK4GRUcell(args).to(args.device)
top_DNN = topDNN(args.top_layers, lastbias = True).to(args.device)

num_sample = scipy.io.loadmat(args.data_path + 'num_sample_select.mat') # 读取返回的是一个dictionary
str_layers = '_'.join(map(str, args.layers))
str_top_layers = '_'.join(map(str, args.top_layers))

path_save_RK4GRUcell = modelsave_path + 'RK4GRUcell_best_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str(hidden_number) + '.pth'
path_save_topDNN = modelsave_path + 'topDNN_best_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str(hidden_number) + '.pth'

# path_save_RK4GRUcell = modelsave_path + 'RK4GRUcell_last_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str(hidden_number) + '.pth'
# path_save_topDNN = modelsave_path + 'topDNN_last_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str(hidden_number) + '.pth'

# path_save_RK4GRUcell = modelsave_path + 'RK4GRUcell_smallest_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str(hidden_number) + '.pth'
# path_save_topDNN = modelsave_path + 'topDNN_smallest_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str(hidden_number) + '.pth'

print(path_save_RK4GRUcell)
print(path_save_topDNN)

RK4GRUcell.load_state_dict(torch.load(path_save_RK4GRUcell, map_location=torch.device('cpu'),weights_only=True))
top_DNN.load_state_dict(torch.load(path_save_topDNN, map_location=torch.device('cpu'),weights_only=True))

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
    SVi_delay_temp = torch.cat( (test_initial_temp[:,:,0::2].repeat(1, 1,args.layers[-1]), test_initial_temp[:,:,1::2].repeat(1, 1,args.layers[-1]) ),-1)        
    T_time = test_t[0:num_sample_once,:,:].to(torch.float32).to(args.device)
    top_DNN_input = torch.zeros(num_sample_once*test_x_xdot.shape[1],args.top_layers[0])

    svj = SVi_delay_temp
    top_DNN_input_temp = torch.cat((svj[:,:,:svj.shape[2]//2],T_time[:,0:1,:],test_exc[iter_id*num_sample_once:(iter_id + 1)*num_sample_once,0:1,:]),-1)
    top_DNN_input_temp = top_DNN_input_temp.squeeze(1)
    top_DNN_input[0:num_sample_once,:] = top_DNN_input_temp
    
    for i in tqdm(range(test_x_xdot.shape[1] - step_delay_X_dX - 1), desc='Predict tracks'):
        exci_delay = test_exc[iter_id*num_sample_once:(iter_id + 1)*num_sample_once,i:(1 + i), :]
        excj = test_exc[iter_id*num_sample_once:(iter_id + 1)*num_sample_once,(i + 1):(i + 2), :]

        svj,_,_,_,_ = RK4GRUcell(SVi_delay_temp,step_delay_X_dX, step_delay_F,exci_delay,excj)
        top_DNN_input_temp = torch.cat((svj[:,:,:svj.shape[2]//2],T_time[:,(i + 1):(i + 2),:],test_exc[iter_id*num_sample_once:(iter_id + 1)*num_sample_once,0:1,:]),-1)
        top_DNN_input_temp = top_DNN_input_temp.squeeze(1)
        top_DNN_input[(i + 1)*num_sample_once:(i + 2)*num_sample_once,:] = top_DNN_input_temp
        SVi_delay_temp = svj
    
    pred_state = top_DNN(top_DNN_input)
    pred_state = pred_state.reshape(test_x_xdot.shape[1],num_sample_once,test_x_xdot.shape[2])
    pred_state = pred_state.transpose(0,1)
    pred_state_final[iter_id*num_sample_once:(iter_id + 1)*num_sample_once,:,:] = pred_state.detach()

####saving data#####
E_X_pred = np.transpose(pred_state_final[:,:,0].numpy(), axes = [1,0])
E_X_pred_dict = {'E_X_pred_'+str(hidden_number):E_X_pred} 
scipy.io.savemat(args.data_path + 'E_X_pred_'+ str(hidden_number)+'.mat', E_X_pred_dict) 

##################plot#####################

dt = args.dt
t = np.linspace(0, pred_state_final.shape[1]-1, pred_state_final.shape[1])*dt
index = 20000
plt.figure(1)
plt.plot(t,test_x_xdot[index,:,0],linestyle = '-',color = 'k')
plt.plot(t,pred_state_final[index,:,0],linestyle = '--',color = 'r')
plt.show()

plt.figure(2)
plt.plot(test_exc[index,:,0],linestyle = '-',color = 'k')
plt.show()

#######loss estimation##########
lamda = args.lamda
Lp = args.Lp
loss_opt = args.loss_opt

###loss of all predicted samples
error_mat = torch.abs(test_x_xdot - pred_state_final)
loss_mse_mat = (error_mat**2)
loss_Lp_mat = (error_mat**Lp)

loss_max = torch.max(error_mat)
loss_mse = torch.mean(loss_mse_mat)
loss_Lp = torch.mean(loss_Lp_mat)

if loss_opt == 0:
    loss_all = ((1 - lamda)*loss_mse + lamda*loss_Lp)
else:
    loss_all = torch.mean(torch.where(error_mat <= 1,loss_mse_mat,2/Lp*loss_Lp_mat+1-2/Lp))
print("loss_mse is ",format(loss_mse))
print("loss_max is ",format(loss_max))
print("loss_all is",format(loss_all))

###loss of the index predicted sample
loss_X_pred_mat = torch.abs(test_x_xdot[index,:,0] - pred_state_final[index,:,0]);
loss_X_pred_mse_mat = loss_X_pred_mat**2;
loss_X_pred_Lp_mat = loss_X_pred_mat**Lp;

loss_X_pred_max = torch.max(loss_X_pred_mat) 
loss_X_pred_mse = torch.mean(loss_X_pred_mse_mat)
loss_X_pred_Lp = torch.mean(loss_X_pred_Lp_mat)

if loss_opt == 0:
    loss_X_pred = ((1 - lamda)*loss_X_pred_mse + lamda*loss_X_pred_Lp)
else:
    loss_X_pred = torch.mean(torch.where(loss_X_pred_mat <= 1,loss_X_pred_mse_mat,2/Lp*loss_X_pred_Lp_mat+1-2/Lp))

print("loss_X_pred_mse is ",format(loss_X_pred_mse))
print("loss_X_pred_max is ",format(loss_X_pred_max))
print("loss_X_pred is ",format(loss_X_pred))

plt.figure(3)
error_X_pred = test_x_xdot[index,:,0] - pred_state_final[index,:,0]
plt.plot(t,error_X_pred,linestyle = '--',color = 'r')
plt.show()

##########training and validation loss##########
##loss
path_train_loss = modelsave_path + 'train_epochs_loss_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str(hidden_number) + '.npy'
path_val_loss = modelsave_path + 'val_epochs_loss_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str(hidden_number) + '.npy'
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

