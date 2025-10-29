function Paper1_force_correlation_sample_simulation;
%%%%%Sample_simulation%%%%%
clear;clc;close all;
current_path = cd;
Data_path = [current_path,'\data\'];
mkdir(Data_path);

fs = 100;
dt = 1/fs;
T = 10 - dt;
t = 0:dt:T;
t(t == 0) = 1e-5;
lt = length(t);

df = fs/lt;
f = df*[-ceil(lt/2)+1:1:floor(lt/2)]';
f(f == 0) = 1e-5;
lf = length(f);


%%%%basis function%%%%%
f_number_sine = 3;
f_number_cosine = 2;
f_number = f_number_sine + f_number_cosine;
f_basis = f(lf/2+2:2:lf/2+2*f_number);
f_basis_sine = f_basis(1:2:end);
f_basis_cosin = f_basis(2:2:end);
Basis_sin = sin(2*pi*f_basis_sine*t)';
Basis_cos = cos(2*pi*f_basis_cosin*t)';

num_sample_all = 5e4;
Acc_std = 100;

% coef_rand_sin1 = randn(f_number_sine,num_sample_all)*sqrt(Acc_std^2/f_number);
% Acc_sin1 = Basis_sin*coef_rand_sin1;
% coef_rand_cos1 = randn(f_number_cosine,num_sample_all)*sqrt(Acc_std^2/f_number);
% Acc_cos1 = Basis_cos*coef_rand_cos1;
% Acc1 = Acc_sin1+Acc_cos1;

coef_rand_sin = (rand(f_number_sine,num_sample_all)-0.5)/0.5*35;
Acc_sin = Basis_sin*coef_rand_sin;
coef_rand_cos = (rand(f_number_cosine,num_sample_all)-0.5)/0.5*35;
Acc_cos = Basis_cos*coef_rand_cos;
Acc = Acc_sin+Acc_cos;

%%%%%%%%%%%%Sample simulation%%%%%%%%%
save([Data_path,'Acc.mat'],'Acc', '-v7.3');

end
