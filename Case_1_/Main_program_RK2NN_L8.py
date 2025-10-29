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
        ### model parameter
        self.dt = 0.01                            
        self.SV_feature = 2                       
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.layers = [10,5]   #M_Gamma*p
        self.top_layers = [self.layers[-1],1]
        self.hidden_number = 1
        self.lamda = 1
        self.Lp = 8
        self.loss_opt = 0 
        
        ### training parameter
        self.batch_size = 2000  
        self.seq_len = 1 
        self.stage1 = 0.025*0;  self.stage2 = 0.05*0    
        self.lr = 0.005
        self.lr_step = 100
        self.lr_gamma = 0.9772
        self.para_number = [91+1,201+2,311+3,421+4,531+5]
        self.beta = (0.9,0.9)
        self.derivative_norm = 1
        
        ### Epoch
        self.epochs = 10000
        self.valper = 1
        self.lr_threshold = 0
        
        ### file path
        self.data_path = r'data/'
        self.modelsave_path = r'Results/'
        if not os.path.exists(self.modelsave_path):
            os.makedirs(self.modelsave_path)

###############################################################################
##function to load data
def load_matdata_train(args):
    data_path_F = args.data_path + 'F_train.mat'
    data_path_X_dX_input_train = args.data_path + 'X_dX_input_train.mat'
    data_path_X_dX_output_train = args.data_path + 'X_dX_output_train.mat'
    data_path_E_X_output_train = args.data_path + 'E_X_output_train.mat'
    data_path_t_train = args.data_path + 't_train.mat'
    
    data_path_coef_F = args.data_path + 'coef_F.mat'
    data_path_coef_X = args.data_path + 'coef_X_output.mat'
    data_path_coef_dX = args.data_path + 'coef_dX_output.mat'
    data_path_coef_ddX = args.data_path + 'coef_ddX_output.mat'
    data_path_coef_E_X = args.data_path + 'coef_E_X_output.mat'
    
    with h5py.File(data_path_F, 'r') as file:
        variable_name = list(file.keys())[0]
        F_input_train = file[variable_name][:]
    F_input_train = F_input_train[:, np.newaxis]
    F_input_train = np.transpose(F_input_train, axes = [2,0,1])
    
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
    
    return F_input_train, X_dX_input_train, X_dX_output_train, E_X_output_train, t_train, coef_F,coef_X,coef_dX,coef_ddX, coef_E_X

##Lp loss
# def p_norm_loss(pred, target, p = 4):
#     return torch.mean(torch.abs(pred - target) ** p)
#     # return torch.pow(torch.mean(torch.abs(pred - target) ** p), 1/p)

###############################################################################
##main function
def train_RK4PIGRU_main(args):
    modelsave_path = args.modelsave_path 
    data_path = args.data_path  
    F_input, X_dX_input, X_dX_output, E_X_output, t_input, coef_F,coef_X,coef_dX,coef_ddX,coef_E_X = load_matdata_train(args) #[number,length,feature]
    
    #################
    indices = np.arange(X_dX_input.shape[0]) 
    # np.random.shuffle(indices)
    
    train_indices = indices[0:round(indices.shape[0]*0.8)]
    val_indices = indices[round(indices.shape[0]*0.8):indices.shape[0]]
    
    F_input_train = torch.from_numpy(F_input[train_indices, :, :])
    X_dX_input_train = torch.from_numpy(X_dX_input[train_indices, :, :])
    X_dX_output_train = torch.from_numpy(X_dX_output[train_indices, :, :])
    E_X_output_train = torch.from_numpy(E_X_output[train_indices, :, :])
    t_input_train = torch.from_numpy(t_input[train_indices, :, :])

    F_input_val = torch.from_numpy(F_input[val_indices, :, :])
    X_dX_input_val = torch.from_numpy(X_dX_input[val_indices, :, :])
    X_dX_output_val = torch.from_numpy(X_dX_output[val_indices, :, :])
    E_X_output_val = torch.from_numpy(E_X_output[val_indices, :, :])
    t_input_val = torch.from_numpy(t_input[val_indices, :, :])

    # val_indices_temp = train_indices[0:train_indices.shape[0]//2]
    # F_input_val = torch.from_numpy(F_input[val_indices_temp, :, :])
    # X_dX_input_val = torch.from_numpy(X_dX_input[val_indices_temp, :, :])
    # X_dX_output_val = torch.from_numpy(X_dX_output[val_indices_temp, :, :])
    # E_X_output_val = torch.from_numpy(E_X_output[val_indices_temp, :, :])
    # t_input_val = torch.from_numpy(t_input[val_indices_temp, :, :])
    #####################
    train_dataset = torch.utils.data.TensorDataset(F_input_train, X_dX_input_train,E_X_output_train,t_input_train)
    val_dataset = torch.utils.data.TensorDataset(F_input_val, X_dX_input_val, E_X_output_val,t_input_val)
      
    train_dataloader = DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = True)
    val_dataloader = DataLoader(dataset = val_dataset, batch_size = args.batch_size, shuffle = False)
  
    ####layers,gru_step, F,X dX,and delay####
    num_input_layer_X_dX = torch.numel(X_dX_input_train[0,:,:])
    num_input_layer_F = len(F_input_train[0,0,:])
    hidden_number = args.hidden_number
    args.layers.insert(0,num_input_layer_F + num_input_layer_X_dX*args.layers[-1])
    args.top_layers[0] += (t_input.shape[2] + num_input_layer_F)
    q = args.top_layers[-1]
    for i in range(hidden_number): 
        args.top_layers.insert(i+1,(args.layers[-1]+num_input_layer_F+4)*q)
    del q
    
    # layers_last = copy.deepcopy(args.layers)
    top_layers_last = copy.deepcopy(args.top_layers)
    del top_layers_last[-2]

    gru_step = torch.numel(E_X_output_train[0,:,0])
    step_delay_F = torch.numel(F_input_train[0,:,0]) - torch.numel(X_dX_output_train[0,:,0])
    step_delay_X_dX = torch.numel(X_dX_input_train[0,:,0]) - 1

    ####neural network, optimization method, and parameters
    RK4GRUcell = myRK4GRUcell(args).to(args.device)
    top_DNN = topDNN(args.top_layers, lastbias = True).to(args.device)

    ############provide the same initial mapping##############
    if hidden_number > 1:
        num_sample = scipy.io.loadmat(args.data_path + 'num_sample_select.mat')
        path_save_RK4GRUcell_last = modelsave_path + 'RK4GRUcell_last_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str(hidden_number-1) + '.pth'
        path_save_topDNN_last = modelsave_path + 'topDNN_last_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str(hidden_number-1) + '.pth'
        
        RK4GRUcell.load_state_dict(torch.load(path_save_RK4GRUcell_last,map_location=args.device,weights_only=True))

        top_DNN_last = topDNN(top_layers_last, lastbias = True)
        top_DNN_last.load_state_dict(torch.load(path_save_topDNN_last,map_location=args.device,weights_only=True))

        for name, layer in top_DNN.layers.named_children():

            if isinstance(layer, torch.nn.BatchNorm1d) and "batchnorm" in name and int(name[-1]) < hidden_number-1:
                with torch.no_grad():
                    layer.running_mean.copy_(top_DNN_last.layers._modules[name].running_mean.clone())
                    layer.running_var.copy_(top_DNN_last.layers._modules[name].running_var.clone())

            if isinstance(layer, torch.nn.Linear) and "mlp_layer" in name and int(name[-1]) < hidden_number-1:
                with torch.no_grad():
                    layer.weight.copy_(top_DNN_last.layers._modules[name].weight.clone())
                    layer.bias.copy_(top_DNN_last.layers._modules[name].bias.clone())
        
            if isinstance(layer,torch.nn.PReLU) and "activation" in name and int(name[-1]) < hidden_number-1:
                with torch.no_grad():
                    layer.weight.copy_(top_DNN_last.layers._modules[name].weight.clone())

            if isinstance(layer, torch.nn.Linear) and "output_layer" in name:    
                with torch.no_grad():
                    layer.weight.copy_(top_DNN_last.layers._modules[name].weight.clone())
                    layer.bias.copy_(top_DNN_last.layers._modules[name].bias.clone())

            if isinstance(layer, torch.nn.Linear) and "mlp_layer" in name and int(name[-1]) == hidden_number-1:    
                with torch.no_grad():
                    layer.weight.zero_()
                    layer.bias.zero_()
                    dim = min(layer.in_features, layer.out_features)
                    layer.weight[:dim, :dim] = torch.eye(dim)

            if isinstance(layer,torch.nn.PReLU) and "activation" in name and int(name[-1]) == hidden_number-1:
                with torch.no_grad():
                    layer.weight.data.fill_(1)
        
        del path_save_RK4GRUcell_last
        del path_save_topDNN_last
        del top_DNN_last
    ###########################################################

    Adam_beta = args.beta
    derivative_norm = args.derivative_norm
    optimizer_RK4GRUcell = torch.optim.Adam(RK4GRUcell.parameters(), lr=args.lr,betas = Adam_beta)
    optimizer_topDNN = torch.optim.Adam(top_DNN.parameters(), lr=args.lr,betas = Adam_beta)
    # optimizer = torch.optim.SGD(RK4GRUcell.parameters(), lr=args.lr)
    
    lr_scheduler_RK4GRUcell = torch.optim.lr_scheduler.StepLR(optimizer_RK4GRUcell,args.lr_step, args.lr_gamma)
    lr_scheduler_topDNN = torch.optim.lr_scheduler.StepLR(optimizer_topDNN,args.lr_step, args.lr_gamma)
    lr_threshold = args.lr_threshold
    top_DNN_derivative_threshold = (args.para_number[hidden_number-1]/args.para_number[0])**0.5

    criterion = torch.nn.MSELoss(reduction = 'none')

    ##Lp loss
    Lp = args.Lp
    lamda = args.lamda
    loss_opt = args.loss_opt

    train_epochs_loss = []
    train_epochs_loss_mse = []
    train_epochs_loss_max = []

    val_epochs_loss = []
    val_epochs_loss_mse = []
    val_epochs_loss_max = []

    smallest_loss = torch.tensor(float('inf'))
    smallest_loss_mse = torch.tensor(float('inf'))
    smallest_loss_max = torch.tensor(float('inf'))
    smallest_epoch = 0
    smallest_epoch_mse = 0
    smallest_epoch_max = 0
    SVj_smallest = 0

    best_loss = torch.tensor(float('inf'))
    best_loss_mse = torch.tensor(float('inf'))
    best_loss_max = torch.tensor(float('inf'))
    best_epoch = 0
    best_epoch_mse = 0
    best_epoch_max = 0

    errors = torch.tensor(float('inf'))
    
    RK4GRUcell_derivative_norm = []
    top_DNN_derivative_norm = []

    ##initial random seeds again###
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    print('Training...')
    print('top_DNN_derivative_threshold is: '+f"{top_DNN_derivative_threshold:.3f}")
    ##training####
    for epoch in range(args.epochs):
        
        start_time = timeit.default_timer()
        RK4GRUcell.train()
        top_DNN.train()

        train_epoch_loss = []
        train_epoch_loss_mse = []
        train_epoch_loss_max = []
        RK4GRUcell_epoch_derivative_norm = []
        top_DNN_epoch_derivative_norm = []
        
        for idx, (Exc,SVi_delay, SVjtarget,T_time) in enumerate(train_dataloader):
               
            Exc = Exc.to(torch.float32).to(args.device)
            SVi_delay = SVi_delay.to(torch.float32).to(args.device)
            SVjtarget = SVjtarget.to(torch.float32).to(args.device)
            T_time = T_time.to(torch.float32).to(args.device)
            top_DNN_input = torch.zeros(SVjtarget.shape[0]*SVjtarget.shape[1],(args.layers[-1]+ Exc.shape[2] + 1)   ).to(args.device)
            
            SV_next = torch.cat( (SVi_delay[:,:,0::2].repeat(1, 1,args.layers[-1]), SVi_delay[:,:,1::2].repeat(1, 1,args.layers[-1]) ),-1)
            top_DNN_input_temp = torch.cat((SV_next[:,:,:SV_next.shape[2]//2],T_time[:,0:1,:],Exc[:,0:1,:]),-1)
            top_DNN_input_temp = top_DNN_input_temp.squeeze(1)
            top_DNN_input[0:SVjtarget.shape[0],:] = top_DNN_input_temp
            SVi_delay_temp = SV_next 

            for gru_s in range(gru_step - 1): 
                exci_delay = Exc[:,gru_s:(gru_s + step_delay_F + 1),:] 
                excj = Exc[:,(gru_s + step_delay_F + 1):(gru_s + step_delay_F + 2),:] 

                SV_next,_,_,_,_ = RK4GRUcell(SVi_delay_temp,step_delay_X_dX, step_delay_F,exci_delay,excj)
                top_DNN_input_temp = torch.cat((SV_next[:,:,:SV_next.shape[2]//2],T_time[:,gru_s+1:gru_s+2,:],Exc[:,0:1,:]),-1).squeeze(1)
                top_DNN_input[(gru_s+1)*SVjtarget.shape[0]:(gru_s+2)*SVjtarget.shape[0],:] = top_DNN_input_temp
                SVi_delay_temp = SV_next
            
            SVj = top_DNN(top_DNN_input)
            SVj = SVj.reshape(SVjtarget.shape[1], SVjtarget.shape[0], SVjtarget.shape[2])
            SVj = SVj.transpose(0, 1)  
           
            # SVj_reshaped = SVj.reshape(SVjtarget.shape[1], SVjtarget.shape[0], SVjtarget.shape[2])  # [1000, 2000, 1]
            # SVj_reshaped = SVj_reshaped.transpose(0, 1) 

            # loss and optimization            
            SVj = SVj.to(torch.float64)
            SVjtarget = SVjtarget.to(torch.float64)
            error_mat = torch.abs(SVj-SVjtarget)
            error_mat_sort, _ = torch.sort(error_mat.flatten())

            loss_mse_mat = (error_mat_sort**2)
            loss_Lp_mat = (error_mat_sort**Lp)

            loss_max = torch.max(error_mat_sort)
            loss_mse = torch.mean(loss_mse_mat)
            loss_Lp = torch.mean(loss_Lp_mat)

            if loss_opt == 0:
                loss = ((1 - lamda)*loss_mse + lamda*loss_Lp)
            else:
                loss = torch.mean(torch.where(error_mat_sort <= 1,loss_mse_mat,2/Lp*loss_Lp_mat+1-2/Lp))
            
            if (loss.item() < smallest_loss):
                smallest_loss = loss.item()
                smallest_epoch = epoch
            
            if (loss_mse.item() < smallest_loss_mse):
                smallest_loss_mse = loss_mse.item()
                smallest_epoch_mse = epoch
            
            if (loss_max.item() < smallest_loss_max):
                smallest_loss_max = loss_max.item()
                smallest_epoch_max = epoch
                RK4GRUcell_model_smallest = copy.deepcopy(RK4GRUcell.state_dict())
                topDNN_model_smallest = copy.deepcopy(top_DNN.state_dict())
                SVj_smallest = SVj.detach().cpu().clone().numpy()
            
            train_epoch_loss.append(loss.cpu().detach().numpy())
            train_epoch_loss_mse.append(loss_mse.item())
            train_epoch_loss_max.append(loss_max.item())      
                
            ###############
            # if epoch == 0 or epoch == 50 or epoch == 100 or epoch == 1000 or epoch == args.epochs-1:
            #     RK4GRUcell_model_smallest = copy.deepcopy(RK4GRUcell.state_dict())
            #     topDNN_model_smallest = copy.deepcopy(top_DNN.state_dict())
            #     SVj_smallest = SVj.detach().numpy()
            #     RK4GRUcell_temp = myRK4GRUcell(args).to(args.device)
            #     top_DNN_temp = topDNN(args.top_layers, lastbias = True).to(args.device)
                 
            #     RK4GRUcell_temp.load_state_dict(RK4GRUcell_model_smallest)
            #     top_DNN_temp.load_state_dict(topDNN_model_smallest)
                
            #     SVi_delay_smallest = SVi_delay.to(torch.float32).to(args.device)
            #     SVjtarget_smallest = SVjtarget.to(torch.float32).to(args.device)
            #     top_DNN_temp_input = torch.zeros(SVjtarget_smallest.shape[0]*SVjtarget_smallest.shape[1],(args.layers[-1]+ Exc.shape[2] + 1)   )
            
            #     SVi_delay_smallest_temp = torch.cat( (SVi_delay_smallest[:,:,0::2].repeat(1, 1,args.layers[-1]), SVi_delay_smallest[:,:,1::2].repeat(1, 1,args.layers[-1]) ),-1)
            #     SV_next_smallest = SVi_delay_smallest_temp
            #     top_DNN_input_temp = torch.cat((SV_next_smallest[:,:,:SV_next_smallest.shape[2]//2],T_time[:,0:1,:],Exc[:,0:1,:]),-1)
            #     top_DNN_input_temp = top_DNN_input_temp.squeeze(1)
            #     top_DNN_temp_input[0:SVjtarget_smallest.shape[0],:] = top_DNN_input_temp
                
            #     for gru_s in range(gru_step - 1): # gru_step = 401 包含SVi_delay的作为初始时刻的1
            #         exci_smallest_delay = Exc[:,gru_s:(gru_s + step_delay_F + 1),:].to(torch.float32).to(args.device) #当前i步荷载
            #         excj_smallest = Exc[:,(gru_s + step_delay_F + 1):(gru_s + step_delay_F + 2),:].to(torch.float32).to(args.device) #下一步j步荷载

            #         SV_next_smallest,_,_,_,_ = RK4GRUcell_temp(SVi_delay_smallest_temp,step_delay_X_dX, step_delay_F,exci_smallest_delay,excj_smallest)
            #         top_DNN_input_temp = torch.cat((SV_next_smallest[:,:,:SV_next_smallest.shape[2]//2],T_time[:,gru_s+1:gru_s+2,:],Exc[:,0:1,:]),-1).squeeze(1)
            #         top_DNN_temp_input[(gru_s+1)*SVjtarget_smallest.shape[0]:(gru_s+2)*SVjtarget_smallest.shape[0],:] = top_DNN_input_temp
            #         SVi_delay_smallest_temp = SV_next_smallest
                
            #     SVj_smallest_temp = top_DNN(top_DNN_temp_input)
            #     SVj_smallest_temp = SVj_smallest_temp.reshape(SVjtarget_smallest.shape[1], SVjtarget_smallest.shape[0], SVjtarget_smallest.shape[2])  # 先分为 [1000, 2000, 1]
            #     SVj_smallest_temp = SVj_smallest_temp.transpose(0, 1)
                
            #     SVj_smallest_temp = SVj_smallest_temp.detach().numpy()
            #     errors = np.max(np.abs(SVj_smallest_temp - SVj_smallest))
                
            ###############
            optimizer_RK4GRUcell.zero_grad()
            optimizer_topDNN.zero_grad()
            loss.backward()
            rk4_norm = torch.nn.utils.clip_grad_norm_(RK4GRUcell.parameters(), derivative_norm)
            top_norm = torch.nn.utils.clip_grad_norm_(top_DNN.parameters(), derivative_norm*top_DNN_derivative_threshold)
            optimizer_RK4GRUcell.step()
            optimizer_topDNN.step()

            RK4GRUcell_epoch_derivative_norm.append(rk4_norm.item())
            top_DNN_epoch_derivative_norm.append(top_norm.item())
        
        RK4GRUcell_derivative_norm.append([epoch, *RK4GRUcell_epoch_derivative_norm])
        top_DNN_derivative_norm.append([epoch, *top_DNN_epoch_derivative_norm])

        train_epochs_loss.append([epoch, np.average(train_epoch_loss)])
        train_epochs_loss_mse.append([epoch, *train_epoch_loss_mse])
        train_epochs_loss_max.append([epoch, *train_epoch_loss_max])

        print('###################### epoch_{} ######################'.format(epoch),flush = True)
        print("[train lr_scheduler_RK4GRUcell = {}]".format( lr_scheduler_RK4GRUcell.get_last_lr()[0]),flush = True)
        print("[train lr_scheduler_topDNN = {}]".format( lr_scheduler_topDNN.get_last_lr()[0]),flush = True)
        print("loss = {}".format(np.average(train_epoch_loss)),flush = True)
        print("temp loss vector = {}".format([float(x) for x in train_epoch_loss]),flush = True)
        print("temp loss_mse vector = {}".format(train_epoch_loss_mse),flush = True)
        print("temp loss_max vector = {}".format(train_epoch_loss_max),flush = True)
        print("RK4GRUcell_epoch_derivative_norm = {}".format(RK4GRUcell_epoch_derivative_norm),flush = True)
        print("top_DNN_epoch_derivative_norm = {}".format(top_DNN_epoch_derivative_norm),flush = True)
        print(' ') 

        if lr_scheduler_RK4GRUcell.get_last_lr()[0] >= lr_threshold:
            lr_scheduler_RK4GRUcell.step()
        
        if lr_scheduler_topDNN.get_last_lr()[0] >= lr_threshold:
            lr_scheduler_topDNN.step()

        if epoch % args.valper == 0 or epoch==args.epochs-1:
            RK4GRUcell.eval()
            top_DNN.eval()
            val_epoch_loss = []
            val_epoch_loss_mse = []
            val_epoch_loss_max = []

            for idx, (Exc,SVi_delay, SVjtarget,T_time) in enumerate(val_dataloader):
                Exc = Exc.to(torch.float32).to(args.device)
                SVi_delay = SVi_delay.to(torch.float32).to(args.device)
                SVjtarget = SVjtarget.to(torch.float32).to(args.device)
                top_DNN_input = torch.zeros(SVjtarget.shape[0]*SVjtarget.shape[1],(args.layers[-1]+ Exc.shape[2] + 1)   ).to(args.device)
                T_time = T_time.to(torch.float32).to(args.device)

                SV_next = torch.cat( (SVi_delay[:,:,0::2].repeat(1, 1,args.layers[-1]), SVi_delay[:,:,1::2].repeat(1, 1,args.layers[-1]) ),-1)
                top_DNN_input_temp = torch.cat((SV_next[:,:,:SV_next.shape[2]//2],T_time[:,0:1,:],Exc[:,0:1,:]),-1)
                top_DNN_input_temp = top_DNN_input_temp.squeeze(1)
                top_DNN_input[0:SVjtarget.shape[0],:] = top_DNN_input_temp
                SVi_delay_temp = SV_next
                   
                for gru_s in range(gru_step - 1): # gru_step = 200
                    exci_delay = Exc[:,gru_s:(gru_s + step_delay_F + 1),:].to(torch.float32).to(args.device) 
                    excj = Exc[:,(gru_s + step_delay_F + 1):(gru_s + step_delay_F + 2),:].to(torch.float32).to(args.device) 
                    SV_next,_,_,_,_ = RK4GRUcell(SVi_delay_temp,step_delay_X_dX, step_delay_F,exci_delay,excj)
                    
                    top_DNN_input_temp = torch.cat((SV_next[:,:,:SV_next.shape[2]//2],T_time[:,gru_s+1:gru_s+2,:],Exc[:,0:1,:]),-1).squeeze(1)
                    top_DNN_input[(gru_s+1)*SVjtarget.shape[0]:(gru_s+2)*SVjtarget.shape[0],:] = top_DNN_input_temp
                    SVi_delay_temp = SV_next
                
                SVj = top_DNN(top_DNN_input)
                SVj = SVj.reshape(SVjtarget.shape[1], SVjtarget.shape[0], SVjtarget.shape[2])
                SVj = SVj.transpose(0, 1)
                
                SVj = SVj.to(torch.float64)
                SVjtarget = SVjtarget.to(torch.float64)
                error_mat = torch.abs(SVj-SVjtarget)
                error_mat_sort, _ = torch.sort(error_mat.flatten())

                loss_mse_mat = (error_mat_sort**2)
                loss_Lp_mat = (error_mat_sort**Lp)

                loss_max = torch.max(error_mat_sort)
                loss_mse = torch.mean(loss_mse_mat)
                loss_Lp = torch.mean(loss_Lp_mat)

                if loss_opt == 0:
                    loss = ((1 - lamda)*loss_mse + lamda*loss_Lp)  
                else:
                    loss = torch.mean(torch.where(error_mat_sort <= 1,loss_mse_mat,2/Lp*loss_Lp_mat+1-2/Lp))  
   
                if (loss.item() < best_loss):
                    best_loss = loss.item()
                    best_epoch = epoch
                
                if (loss_mse.item() < best_loss_mse):
                    best_loss_mse = loss_mse.item()
                    best_epoch_mse = epoch
                
                if (loss_max.item() < best_loss_max):
                    best_loss_max = loss_max.item()
                    best_epoch_max = epoch
                    RK4GRUcell_model_best = copy.deepcopy(RK4GRUcell.state_dict())
                    topDNN_model_best = copy.deepcopy(top_DNN.state_dict())
                
                val_epoch_loss.append(loss.cpu().detach().numpy())
                val_epoch_loss_mse.append(loss_mse.item())
                val_epoch_loss_max.append(loss_max.item())
            
            val_epochs_loss.append([epoch, np.average(val_epoch_loss)])
            val_epochs_loss_mse.append([epoch, *val_epoch_loss_mse]) 
            val_epochs_loss_max.append([epoch, *val_epoch_loss_max])           
            print("[val] loss = {}".format(np.average(val_epoch_loss)),flush = True)
            print("[val] temp loss_mse vector = {}".format(val_epoch_loss_mse),flush = True)
            print("[val] temp loss_max vector = {}".format(val_epoch_loss_max),flush = True)
            print(' ')

        end_time = timeit.default_timer()
        time_consume = end_time - start_time
        print(f'smallest_epoch is {smallest_epoch}',flush = True)
        print(f'smallest_loss is {smallest_loss:.6f}',flush = True)
        print(f'smallest_epoch_mse is {smallest_epoch_mse}',flush = True)
        print(f'smallest_loss_mse is {smallest_loss_mse:.6f}',flush = True)
        print(f'smallest_epoch_max is {smallest_epoch_max}',flush = True)
        print(f'smallest_loss_max is {smallest_loss_max:.6f}',flush = True)
        print('  ')
        print(f'best_epoch is {best_epoch}',flush = True)
        print(f'best_loss is {best_loss:.6f}',flush = True)
        print(f'best_epoch_mse is {best_epoch_mse}',flush = True)
        print(f'best_loss_mse is {best_loss_mse:.6f}',flush = True)
        print(f'best_epoch_max is {best_epoch_max}',flush = True)
        print(f'best_loss_max is {best_loss_max:.6f}',flush = True)
        
        print(f'errors is {errors:.6f}',flush = True)
        print(f'Consumed time is {time_consume:.3f} s',flush = True)
        print(' ')
    ##########################################################################################################
    RK4GRUcell_model_last = RK4GRUcell.state_dict()
    topDNN_model_last = top_DNN.state_dict()

    ###saving trained models###
    num_sample = scipy.io.loadmat(args.data_path + 'num_sample_select.mat')
    str_layers = '_'.join(map(str, args.layers))
    str_top_layers = '_'.join(map(str, args.top_layers))

    path_save_model_best = modelsave_path + 'RK4GRUcell_best_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str(hidden_number) + '.pth'
    torch.save(RK4GRUcell_model_best, path_save_model_best)

    path_save_model_last = modelsave_path + 'RK4GRUcell_last_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str(hidden_number) + '.pth'
    torch.save(RK4GRUcell_model_last, path_save_model_last)
    
    path_save_model_smallest = modelsave_path + 'RK4GRUcell_smallest_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str(hidden_number) + '.pth'
    torch.save(RK4GRUcell_model_smallest, path_save_model_smallest)
   
    path_save_model_best = modelsave_path + 'topDNN_best_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str(hidden_number) + '.pth'
    torch.save(topDNN_model_best, path_save_model_best)

    path_save_model_last = modelsave_path + 'topDNN_last_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str(hidden_number) + '.pth'
    torch.save(topDNN_model_last, path_save_model_last)

    path_save_model_smallest = modelsave_path + 'topDNN_smallest_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str(hidden_number) + '.pth'
    torch.save(topDNN_model_smallest, path_save_model_smallest)
    
    ###saving loss###
    ##loss
    train_epochs_loss = np.array(train_epochs_loss)
    val_epochs_loss = np.array(val_epochs_loss)
    
    path_save_train_epochs_loss = modelsave_path + 'train_epochs_loss_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str(hidden_number) + '.npy'
    path_save_val_epochs_loss = modelsave_path + 'val_epochs_loss_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str(hidden_number) + '.npy'
    np.save(path_save_train_epochs_loss, train_epochs_loss)
    np.save(path_save_val_epochs_loss, val_epochs_loss)
    
    Train_epochs_loss = {'Train_epochs_loss_'+str(hidden_number):train_epochs_loss}
    Val_epochs_loss = {'Val_epochs_loss_'+str(hidden_number):val_epochs_loss}
    scipy.io.savemat(data_path + 'Train_epochs_loss_'+str(hidden_number)+'.mat', Train_epochs_loss)
    scipy.io.savemat(data_path + 'Val_epochs_loss_'+str(hidden_number)+'.mat', Val_epochs_loss)
    
    ##loss_mse
    # train_epochs_loss_mse = np.array(train_epochs_loss_mse)
    # val_epochs_loss_mse = np.array(val_epochs_loss_mse)

    path_save_train_epochs_loss_mse = modelsave_path + 'train_epochs_loss_mse_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str(hidden_number) + '.npy'
    path_save_val_epochs_loss_mse = modelsave_path + 'val_epochs_loss_mse_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str(hidden_number) + '.npy'
    np.save(path_save_train_epochs_loss_mse, train_epochs_loss_mse)
    np.save(path_save_val_epochs_loss_mse, val_epochs_loss_mse)

    Train_epochs_loss_mse = {'Train_epochs_loss_mse_'+str(hidden_number):train_epochs_loss_mse}
    Val_epochs_loss_mse = {'Val_epochs_loss_mse_'+str(hidden_number):val_epochs_loss_mse}
    scipy.io.savemat(data_path + 'Train_epochs_loss_mse_'+str(hidden_number)+'.mat', Train_epochs_loss_mse)
    scipy.io.savemat(data_path + 'Val_epochs_loss_mse_'+str(hidden_number)+'.mat', Val_epochs_loss_mse)

    ##loss_max
    # train_epochs_loss_max = np.array(train_epochs_loss_max)
    # val_epochs_loss_max = np.array(val_epochs_loss_max)

    path_save_train_epochs_loss_max = modelsave_path + 'train_epochs_loss_max_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str(hidden_number) + '.npy'
    path_save_val_epochs_loss_max = modelsave_path + 'val_epochs_loss_max_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str(hidden_number) + '.npy'
    np.save(path_save_train_epochs_loss_max, train_epochs_loss_max)
    np.save(path_save_val_epochs_loss_max, val_epochs_loss_max)

    Train_epochs_loss_max = {'Train_epochs_loss_max_'+str(hidden_number):train_epochs_loss_max}
    Val_epochs_loss_max = {'Val_epochs_loss_max_'+str(hidden_number):val_epochs_loss_max}
    scipy.io.savemat(data_path + 'Train_epochs_loss_max_'+str(hidden_number)+'.mat', Train_epochs_loss_max)
    scipy.io.savemat(data_path + 'Val_epochs_loss_max_'+str(hidden_number)+'.mat', Val_epochs_loss_max)

    ###saving indices
    path_save_train_indices = modelsave_path + 'train_indices_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str(hidden_number) + '.npy'
    path_save_val_indices = modelsave_path + 'val_indices_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str(hidden_number) + '.npy'
    np.save(path_save_train_indices, train_indices)
    np.save(path_save_val_indices, val_indices)

    # X_smallest = np.transpose(SVj_smallest[:,:,0], axes = [1,0])
    # X_smallest = {'X_smallest':X_smallest  } 
    # scipy.io.savemat(data_path + 'X_smallest.mat', X_smallest)

    ###derivative norm
    RK4GRUcell_derivative_norm = np.array(RK4GRUcell_derivative_norm)
    top_DNN_derivative_norm = np.array(top_DNN_derivative_norm)
    
    # path_save_RK4GRUcell_derivative_norm = modelsave_path + 'RK4GRUcell_derivative_norm_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str(hidden_number) + '.npy'
    # path_save_top_DNN_derivative_norm = modelsave_path + 'top_DNN_derivative_norm_' + str(gru_step) + '_' + str(num_sample['num_sample_select'][0,0]) + '_' + str(args.epochs) + '_' + str(hidden_number) + '.npy'
    # np.save(path_save_RK4GRUcell_derivative_norm, RK4GRUcell_derivative_norm)
    # np.save(path_save_top_DNN_derivative_norm, top_DNN_derivative_norm)
    
    RK4GRUcell_derivative_norm_file = {'RK4GRUcell_derivative_norm_'+str(hidden_number):RK4GRUcell_derivative_norm}
    top_DNN_derivative_norm_file = {'top_DNN_derivative_norm_'+str(hidden_number):top_DNN_derivative_norm}
    scipy.io.savemat(data_path + 'RK4GRUcell_derivative_norm_'+str(hidden_number)+'.mat', RK4GRUcell_derivative_norm_file)
    scipy.io.savemat(data_path + 'top_DNN_derivative_norm_'+str(hidden_number)+'.mat', top_DNN_derivative_norm_file)

    return RK4GRUcell_model_last,topDNN_model_last,train_epochs_loss,val_epochs_loss

###############################################################################
###############################################################################
#Training Module

args = Args()
train_RK4PIGRU_main(args)

