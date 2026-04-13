clear;clc;close all;
current_path = cd;
Data_path = [current_path,'\data\'];
load([Data_path,'Acc.mat']);

%%%%%%%%%%%%
[num_time,num_sample] = size(Acc);
dt = 0.001;
t = 0:dt:(num_time-1)*dt;
T = num_time*dt;
t_total = linspace(-2*num_time+1,2*num_time,4*num_time)*dt;
t_total_length = length(t_total);
fs = 1/dt;
df = fs/t_total_length;

%%%%%%%%%%%%%
Acc_zero_initial = (Acc - Acc(1,:))';
Acc_zero_initial_lr = fliplr(Acc_zero_initial);
Acc_zero_initial_max = max(max(abs(Acc_zero_initial)));
Acc_zero_initial_std = mean(mean((Acc_zero_initial).^2)).^0.5;
LK = max(max(abs(diff(Acc)/dt)));

%%%%%%%%%%%%%%%%%
c_K = 1;
M_limit = (10*c_K);
M_limit = max(M_limit,1);
M_number = 10;
M = floor(M_limit*(2.^[0:1:(M_number-1)]'));
M(M>num_time/2) = 5000;
r = floor(log(M));
v = T*r.^(1./(r+1)).*(c_K.^(r./(r+1))).*(factorial(r).^(2./(r+1))).*(log(M).^(1./(r+1)))./(M.^(r./(r+1)));

%%%%%%%%%%%%%%%%%
Eps_sup = zeros(1,M_number);
Eps_L2 = zeros(1,M_number);

for i = 1:M_number
    rou_v = bump_rou_v(t_total,v(i),dt);
    fre_temp = [1:M(i)]*df;
    exp_mat_temp = exp(-2*pi*sqrt(-1)*t_total'*fre_temp);

    F_rou_v = rou_v*exp_mat_temp*dt;
    F_Acc_zero_initial_lr = 2*sqrt(-1)*imag(Acc_zero_initial_lr*exp_mat_temp(t_total_length/2:1:(t_total_length/2+t_total_length/4-1),:)*dt);
    Acc_zero_initial_lr_estimate = real((F_Acc_zero_initial_lr.*F_rou_v)*exp_mat_temp(t_total_length/2:1:(t_total_length/2+t_total_length/4-1),:)'/2/T);

    Eps_sup(i) = max(max(abs(Acc_zero_initial_lr_estimate - Acc_zero_initial_lr)))/Acc_zero_initial_max;
    Eps_L2(i) = mean(mean((Acc_zero_initial_lr_estimate - Acc_zero_initial_lr).^2)).^0.5/Acc_zero_initial_std;
end

%%%%%%%%%plot result%%%%%%%%%
a1 = 0;
b1 = 5;
c1 = -1;
y1 = a1 + b1*log(M).^2.*M.^c1; 
figure(1)
loglog(M,Eps_sup,'o-',M,y1,'*-')

a2 = 0;
b2 = 5;
c2 = -1;
y2 = a2 + b2*log(M).^2.*M.^c2; 
figure(2)
loglog(M,Eps_L2,'o-',M,y2,'*-')
