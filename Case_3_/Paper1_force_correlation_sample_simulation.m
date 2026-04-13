function Paper1_force_correlation_sample_simulation;
%%%%%Sample_simulation%%%%%
clear;clc;close all;
current_path = cd;
Data_path = [current_path,'\data\'];
mkdir(Data_path);
rng(1228);

fs = 1000;
dt = 1/fs;
T = 10 - dt;
t = 0:dt:T;
t(t == 0) = 1e-5;
lt = length(t);

df = fs/lt;
f = df*[-ceil(lt/2)+1:1:floor(lt/2)]';
f(f == 0) = 1e-5;
lf = length(f);

%%%%%%%%%%%%Sample simulation%%%%%%%%%
num_samples = 1000;                         % GP sample paths
ell         = 0.1;                          % GP length-scale

%   k(s,t) = exp( -2 sin^2(pi*(s-t)/T) / ell^2 )
%   Sample paths are C-infinity and hence Lipschitz on [-pi, pi].
D    = sin((1/T) * (t - t'));              % N x N
Kcov = exp(-2 * D.^2 / ell^2) + 1e-8 * eye(lt);
Lch  = chol(Kcov, 'lower');                % Cholesky factor

Acc = (Lch * randn(lt, num_samples));       % N x num_samples

save([Data_path,'Acc.mat'],'Acc', '-v7.3');

end
