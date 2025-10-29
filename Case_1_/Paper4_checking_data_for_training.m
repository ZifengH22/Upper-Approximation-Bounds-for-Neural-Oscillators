clear;clc;
% close all;
current_path = cd;
Data_path = [current_path,'\data\'];
%%%%%%%%%%
load M_response.mat;
load K_response.mat;
load C_response.mat;
load K_in_response.mat;
beta = 2;
gamma_r = 2;
s = 3;
lamda = 0.01;

load([Data_path,'t_train.mat']);
load([Data_path,'F_train.mat']);
load([Data_path,'X_dX_input_train.mat']);
load([Data_path,'X_dX_output_train.mat']);
load([Data_path,'E_X_output_train.mat']);

%%%%%%checking data%%%%%%%%%
t = t_train(1,:);
dt = t(2)-t(1);
[X_check,dX_check,ddX_check,Z_check,~] = Y_response(t,dt,M,C,K_in,beta,gamma_r,s,lamda,F_train');
E_X_check = Ex_extreme_value_process(X_check);

index = 45;
figure(1)
plot(t,X_dX_output_train(index,:),'bo-',t,X_check(:,index),'k--');
xlabel('Time (s)');
ylabel('Response X');
legend('Training data','Checking data');
figure(2)
plot(t,abs(X_dX_output_train(index,:)),'bo-',t,E_X_check(:,index),'k--');
xlabel('Time (s)');
ylabel('Response X');
legend('Training data','Checking data');
