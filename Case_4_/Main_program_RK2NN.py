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

import timeit
import h5py
import copy

seed = 1228
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
###############################################################################
class Args: 
    def __init__(self) -> None:  
        
        self.dt = 0.005                             
        self.SV_feature = 2                        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.layer_width = 2
        self.layers = [self.layer_width,20] 
        self.top_layers = [self.layers[-1],self.layer_width,1]
       
        self.batch_size = 1600  
        self.seq_len = 1  
        
        self.lr = 0.02
        self.lr_step = 100
        self.lr_gamma = 0.96
        self.beta = (0.9,0.99)
        
        self.epochs = 10000
        self.valper = 1
        
        ### file path
        self.data_path = r'data/'
        self.modelsave_path = r'Results/'
        if not os.path.exists(self.modelsave_path):
            os.makedirs(self.modelsave_path)

###############################################################################
def load_matdata_train(args):
    
    data_path_Acc_train = args.data_path + 'Acc_train.mat'
    data_path_X_dX_input_train = args.data_path + 'X_dX_input_train.mat'
    data_path_X_dX_output_train = args.data_path + 'X_dX_output_train.mat'
    data_path_t_train = args.data_path + 't_train.mat'
    
    with h5py.File(data_path_Acc_train, 'r') as file:
        variable_name = list(file.keys())[0]
        Acc_train = file[variable_name][:]
    Acc_train = Acc_train[:, np.newaxis]
    Acc_train = np.transpose(Acc_train, axes = [2,0,1])

    with h5py.File(data_path_X_dX_input_train, 'r') as file:
        variable_name = list(file.keys())[0]
        X_dX_input_train = file[variable_name][:]
    # X_dX_input_train = X_dX_input_train[:, np.newaxis] 
    X_dX_input_train = np.transpose(X_dX_input_train, axes = [2,1,0])

    with h5py.File(data_path_X_dX_output_train, 'r') as file:
        variable_name = list(file.keys())[0]
        X_dX_output_train = file[variable_name][:]
    X_dX_output_train = X_dX_output_train[:, np.newaxis] 
    X_dX_output_train = np.transpose(X_dX_output_train, axes = [2,0,1])

    with h5py.File(data_path_t_train, 'r') as file:
        variable_name = list(file.keys())[0]
        t_train = file[variable_name][:]
    t_train = t_train[:, np.newaxis]
    t_train = np.transpose(t_train, axes = [2,0,1]) 
     
    return Acc_train,X_dX_input_train,X_dX_output_train,t_train

###############################################################################
def train_RK4PIGRU_main(args):
    modelsave_path = args.modelsave_path 
    data_path = args.data_path
    Acc_input,X_dX_input,X_dX_output,t_input = load_matdata_train(args) #[number,length,feature]
    num_sample = scipy.io.loadmat(args.data_path + 'num_sample_select.mat')
    
    #################
    indices = np.arange(Acc_input.shape[0]) 
    # np.random.shuffle(indices)
    
    train_indices = indices[0:round(indices.shape[0]*0.8)]
    val_indices = indices[round(indices.shape[0]*0.8):indices.shape[0]]
    
    Acc_input_train = torch.from_numpy(Acc_input[train_indices, :, :])
    X_dX_input_train = torch.from_numpy(X_dX_input[train_indices, :, :])
    X_dX_output_train = torch.from_numpy(X_dX_output[train_indices, :, :])
    t_input_train = torch.from_numpy(t_input[train_indices, :, :])
 
    Acc_input_val = torch.from_numpy(Acc_input[val_indices, :, :])
    X_dX_input_val = torch.from_numpy(X_dX_input[val_indices, :, :])
    X_dX_output_val = torch.from_numpy(X_dX_output[val_indices, :, :])
    t_input_val = torch.from_numpy(t_input[val_indices, :, :])    
    
    #####################
    train_dataset = torch.utils.data.TensorDataset(Acc_input_train,X_dX_input_train, X_dX_output_train,t_input_train)
    val_dataset = torch.utils.data.TensorDataset(Acc_input_val,X_dX_input_val, X_dX_output_val,t_input_val)
    
    train_dataloader = DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = False)
    val_dataloader = DataLoader(dataset = val_dataset, batch_size = args.batch_size, shuffle = False)
    
    ####layers, Acc, X, dX####
    num_input_layer_Acc = len(Acc_input_train[0,0,:])
    num_input_layer_X_dX = 2*num_input_layer_Acc
    gru_step = torch.numel(Acc_input_train[0,:,0])

    args.layers.insert(0,num_input_layer_Acc + num_input_layer_X_dX*args.layers[-1])
    args.top_layers[0] += (t_input.shape[2] + Acc_input_train.shape[2])
    
    
    ####neural network, optimization method, and parameters
    RK4GRUcell = myRK4GRUcell(args).to(args.device)
    top_DNN = topDNN(args.top_layers, lastbias = True).to(args.device)

    Adam_beta = args.beta
    optimizer_RK4GRUcell = torch.optim.Adam(RK4GRUcell.parameters(), lr=args.lr,betas = Adam_beta)
    optimizer_top_DNN = torch.optim.Adam(top_DNN.parameters(), lr=args.lr,betas = Adam_beta)
    
    lr_scheduler_RK4GRUcell = torch.optim.lr_scheduler.StepLR(optimizer_RK4GRUcell,args.lr_step, args.lr_gamma)
    lr_scheduler_top_DNN = torch.optim.lr_scheduler.StepLR(optimizer_top_DNN,args.lr_step, args.lr_gamma)

    train_epochs_loss = []
    train_epochs_loss_max = []

    val_epochs_loss = []
    val_epochs_loss_max = []
    
    smallest_loss = torch.tensor(float('inf'))
    smallest_loss_max = torch.tensor(float('inf'))
    smallest_epoch = 0
    smallest_epoch_max = 0

    best_loss = torch.tensor(float('inf'))
    best_loss_max = torch.tensor(float('inf'))
    best_epoch = 0
    best_epoch_max = 0

    RK4GRUcell_derivative_norm = []
    top_DNN_derivative_norm = []

    ##initial random seeds again###
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    ##training####
    for epoch in range(args.epochs):
        
        start_time = timeit.default_timer()
        RK4GRUcell.train()
        top_DNN.train()

        train_epoch_loss = []
        train_epoch_loss_max = []
        RK4GRUcell_epoch_derivative_norm = []
        top_DNN_epoch_derivative_norm = []

        for idx, (Exc,SVi_delay, SVjtarget,T_time) in enumerate(train_dataloader):            
            Exc = Exc.to(torch.float32).to(args.device)
            SVi_delay = SVi_delay.to(torch.float32).to(args.device)
            SVjtarget = SVjtarget.to(torch.float32).to(args.device)
            SVj = torch.zeros((SVjtarget.shape[0],SVjtarget.shape[1],SVjtarget.shape[2])).to(torch.float32).to(args.device)
            T_time = T_time.to(torch.float32).to(args.device)
            
            SV_next = torch.cat( (SVi_delay[:,:,0::2].repeat(1, 1,args.layers[-1]), SVi_delay[:,:,1::2].repeat(1, 1,args.layers[-1]) ),-1)
            top_DNN_input = torch.cat((SV_next[:,:,:SV_next.shape[2]//2],T_time[:,0:1,:],Exc[:,0:1,:]),-1)
            SVj[:,0:1,:] = top_DNN(top_DNN_input)
            SVi_delay_temp = SV_next 

            for gru_s in range(gru_step - 1): 
                exci_delay = Exc[:,gru_s:(gru_s + 1),:] 
                excj = Exc[:,(gru_s + 1):(gru_s + 2),:] 

                SV_next,_,_,_,_ = RK4GRUcell(SVi_delay_temp,0.0, 0.0,exci_delay,excj)
                top_DNN_input = torch.cat((SV_next[:,:,:SV_next.shape[2]//2],T_time[:,gru_s+1:gru_s+2,:],Exc[:,0:1,:]),-1)
                SVj[:,gru_s+1:gru_s+2,:] = top_DNN(top_DNN_input)
                SVi_delay_temp = SV_next

            ###############
            error_mat = torch.abs(SVj-SVjtarget)
            error_mat_sort, _ = torch.sort(error_mat.flatten())

            loss_max = torch.max(error_mat_sort)
            loss = torch.mean(error_mat_sort**2)

            if (loss.item() < smallest_loss):
                smallest_loss = loss.item()
                smallest_epoch = epoch
                RK4GRUcell_model_smallest = copy.deepcopy(RK4GRUcell.state_dict())
                top_DNN_model_smallest = copy.deepcopy(top_DNN.state_dict())
            
            if (loss_max.item() < smallest_loss_max):
                smallest_loss_max = loss_max.item()
                smallest_epoch_max = epoch
                RK4GRUcell_model_smallest_max = copy.deepcopy(RK4GRUcell.state_dict())
                top_DNN_model_smallest_max = copy.deepcopy(top_DNN.state_dict())
            
            train_epoch_loss.append(loss.cpu().detach().numpy())
            train_epoch_loss_max.append(loss_max.item())
            #################

            optimizer_RK4GRUcell.zero_grad()
            optimizer_top_DNN.zero_grad()
            loss.backward()
            rk4_norm = torch.nn.utils.clip_grad_norm_(RK4GRUcell.parameters(), 1)
            top_DNN_norm = torch.nn.utils.clip_grad_norm_(top_DNN.parameters(), 1)
            optimizer_RK4GRUcell.step()
            optimizer_top_DNN.step()

            RK4GRUcell_epoch_derivative_norm.append(rk4_norm.item())
            top_DNN_epoch_derivative_norm.append(top_DNN_norm.item())
            
        RK4GRUcell_derivative_norm.append([epoch, *RK4GRUcell_epoch_derivative_norm])
        top_DNN_derivative_norm.append([epoch, *top_DNN_epoch_derivative_norm])

        train_epochs_loss.append([epoch, np.average(train_epoch_loss)])
        train_epochs_loss_max.append([epoch, *train_epoch_loss_max])

        print('###################### epoch_{} ######################'.format(epoch),flush = True)
        print("[train lr_scheduler_RK4GRUcell = {}]".format( lr_scheduler_RK4GRUcell.get_last_lr()[0]),flush = True)
        print("[train lr_scheduler_top_DNN = {}]".format( lr_scheduler_top_DNN.get_last_lr()[0]),flush = True)
        print("loss = {}".format(np.average(train_epoch_loss)),flush = True)
        print("temp loss vector = {}".format([float(x) for x in train_epoch_loss]),flush = True)
        print("temp loss_max vector = {}".format(train_epoch_loss_max),flush = True)
        print("RK4GRUcell_epoch_derivative_norm = {}".format(RK4GRUcell_epoch_derivative_norm),flush = True)
        print("top_DNN_epoch_derivative_norm = {}".format(top_DNN_epoch_derivative_norm),flush = True)
        print(' ') 

        lr_scheduler_RK4GRUcell.step()
        lr_scheduler_top_DNN.step()

        ##########validation##########
        RK4GRUcell.eval()
        top_DNN.eval()
        val_epoch_loss = []
        val_epoch_loss_max = []

        for idx, (Exc,SVi_delay, SVjtarget,T_time) in enumerate(val_dataloader):            
            Exc = Exc.to(torch.float32).to(args.device)
            SVi_delay = SVi_delay.to(torch.float32).to(args.device)
            SVjtarget = SVjtarget.to(torch.float32).to(args.device)
            SVj = torch.zeros((SVjtarget.shape[0],SVjtarget.shape[1],SVjtarget.shape[2])).to(torch.float32).to(args.device)
            T_time = T_time.to(torch.float32).to(args.device)
            
            SV_next = torch.cat( (SVi_delay[:,:,0::2].repeat(1, 1,args.layers[-1]), SVi_delay[:,:,1::2].repeat(1, 1,args.layers[-1]) ),-1)
            top_DNN_input = torch.cat((SV_next[:,:,:SV_next.shape[2]//2],T_time[:,0:1,:],Exc[:,0:1,:]),-1)
            SVj[:,0:1,:] = top_DNN(top_DNN_input)
            SVi_delay_temp = SV_next 

            for gru_s in range(gru_step - 1): 
                exci_delay = Exc[:,gru_s:(gru_s + 1),:] 
                excj = Exc[:,(gru_s + 1):(gru_s + 2),:] 

                SV_next,_,_,_,_ = RK4GRUcell(SVi_delay_temp,0.0, 0.0,exci_delay,excj)
                top_DNN_input = torch.cat((SV_next[:,:,:SV_next.shape[2]//2],T_time[:,gru_s+1:gru_s+2,:],Exc[:,0:1,:]),-1)
                SVj[:,gru_s+1:gru_s+2,:] = top_DNN(top_DNN_input)
                SVi_delay_temp = SV_next
            
            ##########
            error_mat = torch.abs(SVj-SVjtarget)
            error_mat_sort, _ = torch.sort(error_mat.flatten())

            loss_max = torch.max(error_mat_sort)
            loss = torch.mean(error_mat_sort**2)

            if (loss.item() < best_loss):
                best_loss = loss.item()
                best_epoch = epoch
                RK4GRUcell_model_best = copy.deepcopy(RK4GRUcell.state_dict())
                top_DNN_model_best = copy.deepcopy(top_DNN.state_dict())
            
            if (loss_max.item() < best_loss_max):
                best_loss_max = loss_max.item()
                best_epoch_max = epoch
                RK4GRUcell_model_best_max = copy.deepcopy(RK4GRUcell.state_dict())
                top_DNN_model_best_max = copy.deepcopy(top_DNN.state_dict())
            
            val_epoch_loss.append(loss.cpu().detach().numpy())
            val_epoch_loss_max.append(loss_max.item())
        
        val_epochs_loss.append([epoch, np.average(val_epoch_loss)]) 
        val_epochs_loss_max.append([epoch, *val_epoch_loss_max])           
        print("[val] loss = {}".format(np.average(val_epoch_loss)),flush = True)
        print("[val] temp loss_max vector = {}".format(val_epoch_loss_max),flush = True)
        print(' ')

        end_time = timeit.default_timer()
        time_consume = end_time - start_time
        print(f'smallest_epoch is {smallest_epoch}',flush = True)
        print(f'smallest_loss is {smallest_loss:.6f}',flush = True)
        print(f'smallest_epoch_max is {smallest_epoch_max}',flush = True)
        print(f'smallest_loss_max is {smallest_loss_max:.6f}',flush = True)
        print('  ')
        print(f'best_epoch is {best_epoch}',flush = True)
        print(f'best_loss is {best_loss:.6f}',flush = True)
        print(f'best_epoch_max is {best_epoch_max}',flush = True)
        print(f'best_loss_max is {best_loss_max:.6f}',flush = True)
        print(' ')
        print(f'Consumed time is {time_consume:.3f} s',flush = True)
        print(' ')

        #############
        if epoch == 4999 or epoch == 7999:
            RK4GRUcell_model_last = RK4GRUcell.state_dict()
            top_DNN_model_last = top_DNN.state_dict()
   
            str_layers = '_'.join(map(str, args.layers))
            str_top_layers = '_'.join(map(str, args.top_layers))

            path_save_model_best = modelsave_path + 'RK4GRUcell_best_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(int(epoch+1)) + '_' + str_layers + '_' + '.pth'
            torch.save(RK4GRUcell_model_best, path_save_model_best)

            path_save_model_best_max = modelsave_path + 'RK4GRUcell_best_max_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(int(epoch+1)) + '_' + str_layers + '_' + '.pth'
            torch.save(RK4GRUcell_model_best_max, path_save_model_best_max)

            path_save_model_last = modelsave_path + 'RK4GRUcell_last_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(int(epoch+1)) + '_' + str_layers + '_' + '.pth'
            torch.save(RK4GRUcell_model_last, path_save_model_last)
            
            path_save_model_smallest = modelsave_path + 'RK4GRUcell_smallest_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(int(epoch+1)) + '_' + str_layers + '_' + '.pth'
            torch.save(RK4GRUcell_model_smallest, path_save_model_smallest)

            path_save_model_smallest_max = modelsave_path + 'RK4GRUcell_smallest_max_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(int(epoch+1)) + '_' + str_layers + '_' + '.pth'
            torch.save(RK4GRUcell_model_smallest_max, path_save_model_smallest_max)
        

            #####top_DNN_model
            path_save_model_best = modelsave_path + 'top_DNN_model_best_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(int(epoch+1)) + '_' + str_layers + '_' + '.pth'
            torch.save(top_DNN_model_best, path_save_model_best)

            path_save_model_best_max = modelsave_path + 'top_DNN_model_best_max_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(int(epoch+1)) + '_' + str_layers + '_' + '.pth'
            torch.save(top_DNN_model_best_max, path_save_model_best_max)

            path_save_model_last = modelsave_path + 'top_DNN_model_last_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(int(epoch+1)) + '_' + str_layers + '_' + '.pth'
            torch.save(top_DNN_model_last, path_save_model_last)

            path_save_model_smallest = modelsave_path + 'top_DNN_model_smallest_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(int(epoch+1)) + '_' + str_layers + '_' + '.pth'
            torch.save(top_DNN_model_smallest, path_save_model_smallest)

            path_save_model_smallest_max = modelsave_path + 'top_DNN_model_smallest_max_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(int(epoch+1)) + '_' + str_layers + '_' + '.pth'
            torch.save(top_DNN_model_smallest_max, path_save_model_smallest_max)

        ############
    ##########################################################################################################
    RK4GRUcell_model_last = RK4GRUcell.state_dict()
    top_DNN_model_last = top_DNN.state_dict()
    
    num_sample = scipy.io.loadmat(args.data_path + 'num_sample_select.mat')
    str_layers = '_'.join(map(str, args.layers))
    
    #####RK4GRUcell_model
    path_save_model_best = modelsave_path + 'RK4GRUcell_best_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + '.pth'
    torch.save(RK4GRUcell_model_best, path_save_model_best)

    path_save_model_best_max = modelsave_path + 'RK4GRUcell_best_max_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + '.pth'
    torch.save(RK4GRUcell_model_best_max, path_save_model_best_max)

    path_save_model_last = modelsave_path + 'RK4GRUcell_last_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + '.pth'
    torch.save(RK4GRUcell_model_last, path_save_model_last)
    
    path_save_model_smallest = modelsave_path + 'RK4GRUcell_smallest_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + '.pth'
    torch.save(RK4GRUcell_model_smallest, path_save_model_smallest)

    path_save_model_smallest_max = modelsave_path + 'RK4GRUcell_smallest_max_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + '.pth'
    torch.save(RK4GRUcell_model_smallest_max, path_save_model_smallest_max)
   

    #####top_DNN_model
    path_save_model_best = modelsave_path + 'top_DNN_model_best_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + '.pth'
    torch.save(top_DNN_model_best, path_save_model_best)

    path_save_model_best_max = modelsave_path + 'top_DNN_model_best_max_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + '.pth'
    torch.save(top_DNN_model_best_max, path_save_model_best_max)

    path_save_model_last = modelsave_path + 'top_DNN_model_last_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + '.pth'
    torch.save(top_DNN_model_last, path_save_model_last)

    path_save_model_smallest = modelsave_path + 'top_DNN_model_smallest_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + '.pth'
    torch.save(top_DNN_model_smallest, path_save_model_smallest)

    path_save_model_smallest_max = modelsave_path + 'top_DNN_model_smallest_max_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str_layers + '_' + '.pth'
    torch.save(top_DNN_model_smallest_max, path_save_model_smallest_max)

    ############
     ###saving loss###
    ##loss
    train_epochs_loss = np.array(train_epochs_loss)
    val_epochs_loss = np.array(val_epochs_loss)
    
    path_save_train_epochs_loss = modelsave_path + 'train_epochs_loss_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str(args.layer_width) + '.npy'
    path_save_val_epochs_loss = modelsave_path + 'val_epochs_loss_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str(args.layer_width) + '.npy'
    np.save(path_save_train_epochs_loss, train_epochs_loss)
    np.save(path_save_val_epochs_loss, val_epochs_loss)
    
    Train_epochs_loss = {'Train_epochs_loss_'+str(args.layer_width):train_epochs_loss}
    Val_epochs_loss = {'Val_epochs_loss_'+str(args.layer_width):val_epochs_loss}
    scipy.io.savemat(data_path + 'Train_epochs_loss_'+str(args.layer_width)+'.mat', Train_epochs_loss)
    scipy.io.savemat(data_path + 'Val_epochs_loss_'+str(args.layer_width)+'.mat', Val_epochs_loss)
    

    ##loss_max
    train_epochs_loss_max = np.array(train_epochs_loss_max)
    val_epochs_loss_max = np.array(val_epochs_loss_max)

    path_save_train_epochs_loss_max = modelsave_path + 'train_epochs_loss_max_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str(args.layer_width) + '.npy'
    path_save_val_epochs_loss_max = modelsave_path + 'val_epochs_loss_max_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str(args.layer_width) + '.npy'
    np.save(path_save_train_epochs_loss_max, train_epochs_loss_max)
    np.save(path_save_val_epochs_loss_max, val_epochs_loss_max)

    Train_epochs_loss_max = {'Train_epochs_loss_max_'+str_layers:train_epochs_loss_max}
    Val_epochs_loss_max = {'Val_epochs_loss_max_'+str_layers:val_epochs_loss_max}
    scipy.io.savemat(data_path + 'Train_epochs_loss_max_'+str(args.layer_width)+'.mat', Train_epochs_loss_max)
    scipy.io.savemat(data_path + 'Val_epochs_loss_max_'+str(args.layer_width)+'.mat', Val_epochs_loss_max)


    ###derivative norm
    RK4GRUcell_derivative_norm = np.array(RK4GRUcell_derivative_norm)
    top_DNN_derivative_norm = np.array(top_DNN_derivative_norm)
     
    RK4GRUcell_derivative_norm_file = {'RK4GRUcell_derivative_norm_'+str(args.layer_width):RK4GRUcell_derivative_norm}
    top_DNN_derivative_norm_file = {'top_DNN_derivative_norm_'+str(args.layer_width):top_DNN_derivative_norm}
    scipy.io.savemat(data_path + 'RK4GRUcell_derivative_norm_'+str(args.layer_width)+'.mat', RK4GRUcell_derivative_norm_file)
    scipy.io.savemat(data_path + 'top_DNN_derivative_norm_'+str(args.layer_width)+'.mat', top_DNN_derivative_norm_file) 

    return RK4GRUcell_model_last,top_DNN_model_last


#Training Module
args = Args()
train_RK4PIGRU_main(args)

