function Paper2_structural_response_sample_simulation;
%直接计算
clear;clc;close all;
current_path = cd;
Data_path = [current_path,'\data\'];

%%%%%%%%parameter time and frequency%%%%%%%%%%%
fs = 100;
dt = 1/fs;
lt = 1000;
T = lt*dt - dt;
t = 0:dt:T;
t(t == 0) = 1e-5;

lf = lt;
df = fs/lf;
f = [-(0.5*lf - 1):(0.5*lf)]'*df;
f(f == 0) = 1e-5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_degree = 5;%自由度
m = 0.004*345.6*1000;%每层楼的质量(kg)
k = 0.5*170*10^4;%层间位移刚度(N/m)
m_vector = linspace(m,m,num_degree);
k_vector = linspace(2*k,2*k,num_degree);

M = zeros(num_degree,num_degree);
K = zeros(num_degree,num_degree);
C = zeros(num_degree,num_degree);
M = diag(m_vector);%质量矩阵
K = diag(k_vector + [k_vector(2:end),0]) - diag(k_vector(2:end),1) - diag(k_vector(2:end),-1);%刚度矩阵

% R = [zeros(size(M));M^-1];
[Fai,Wnsqr] = eig(K/M,M/M);

Mn = diag(Fai'*M*Fai);
Kn = diag(Fai'*K*Fai);
Wn = diag(Wnsqr.^0.5);

Znb = ones(1,num_degree)'*0.05;
Znb_mat = diag(Znb);
Cn = Znb.*(2*Mn.*Wn);
Cn_mat = diag(Cn);
C = Fai*Cn_mat*Fai';

Cn = diag(Fai'*C*Fai);
Znb = Cn./(2*Mn.*Wn);
fn = Wn.*sqrt(1-Znb.^2)/2/pi


K_in = zeros(size(K));
for ii = 1:num_degree
    if ii ~= num_degree
        K_in(ii,ii:ii+1) = [k_vector(ii),-k_vector(ii + 1)];
    else
        K_in(ii,ii) = k_vector(ii);
    end
end

beta = 2;
gamma_r = 2;
s = 3;
lamda = 0.01;

save M_response.mat M;
save K_response.mat K;
save C_response.mat C;
save K_in_response.mat K_in;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%response1 sample simulation
load([Data_path,'Acc.mat'],'Acc');
[X1,dX1,ddX1,Z1,t_sim1] = Y_response(t,dt,M,C,K_in,beta,gamma_r,s,lamda,Acc(:,:));
E_X1 = Ex_extreme_value_process(X1);

save([Data_path,'X1_response.mat'],'X1', '-v7.3');
save([Data_path,'dX1_response.mat'],'dX1', '-v7.3');
save([Data_path,'ddX1_response.mat'],'ddX1', '-v7.3');
save([Data_path,'Z1_response.mat'],'Z1', '-v7.3');
save([Data_path,'E_X1_response.mat'],'E_X1', '-v7.3');
save time_response_sim1.mat t_sim1;

end


