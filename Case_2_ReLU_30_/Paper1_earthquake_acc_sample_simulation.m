function Paper1_earthquake_acc_sample_simulation
clear;clc;close all;
opt_fig = 0;
current_path = cd;
Data_path = [current_path,'\data\']
mkdir(Data_path);

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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%WVS Loeve spectrum and correlation of earthquake ground motion acceleration%%%%%
[S,R,WVS] = Loevespectrum_WVS_and_correlation(t,f,[2500,0.3]);
% WVS = Kanai_Tajimi_spectrum(t,f,0.1);
% % R_inv = inv(R+1e-1);
% Ematrix = exp(-sqrt(-1)*2*pi*t'*f');
% RS = real(Ematrix*S*Ematrix'*df*df);
% RS_inv = inv(RS);

[V,D] = eig(R);
D_diag = diag(D);
D_diag(abs(D_diag) < 1e-10) = 1e-10;
D_diag(D_diag < 0) = 1e-10;
D = diag(D_diag);
R_rec = V*D*V';
R = (R_rec + R_rec')/2;

R_acc1 = R;
R_acc_sqrt1 = chol(R_acc1)';

save R_acc1.mat R_acc1;
save R_acc_sqrt1.mat R_acc_sqrt1;


%%%%%%%%%Sample simulation%%%%%%%%%%%%;
num_sample_all = 5e4;
Erandom = randn(lt,num_sample_all);
Acc = R_acc_sqrt1*Erandom;        
save([Data_path,'Acc.mat'],'Acc', '-v7.3');

end

