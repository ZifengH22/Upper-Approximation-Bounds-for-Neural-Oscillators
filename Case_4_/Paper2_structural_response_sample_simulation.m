function Paper2_structural_response_sample_simulation_4RK;
clear;clc;close all;
current_path = cd;
Data_path = [current_path,'\data\'];
load([Data_path,'\Acc.mat']);


%% 1. Parameter Definition
N = 200;                    % Number of Degrees of Freedom (DOF)
dt = 0.005;                 % Time step (s) - Reviewer requirement
T_total = 20;               % Total simulation duration (s)
t = 0:dt:(T_total-dt);           % Time vector
Nt = length(t);

% Structural Physical Parameters
% m = 100;                    % Mass per floor (kg)
% k_linear = 8e6;             % Linear stiffness per floor (N/m)
% beta = 4e7;                % Duffing cubic nonlinearity coefficient (N/m^3)

m = 1000;                    % Mass per floor (kg)
k_linear = 1e7;             % Linear stiffness per floor (N/m)
beta = 1e7;                % Duffing cubic nonlinearity coefficient (N/m^3)

[~,number_sample] = size(Acc);

% Construct Mass Matrix (Diagonal)
M_diag = m * ones(N, 1);
M = spdiags(M_diag, 0, N, N);

% Construct Linear Stiffness Matrix (Shear building model)
K_main = 2 * k_linear * ones(N, 1); 
K_main(N) = k_linear;
K_sub = -k_linear * ones(N, 1);
K_sub(N) = 0;
K_lin = spdiags([K_sub K_main K_sub], -1:1, N, N);

% Rayleigh Damping: C = a0*M + a1*K
a0 = 0.02; a1 = 0.002;
C = a0 * M + a1 * K_lin;

%% 2. RK2 Method Integration Loop
X1 = zeros(number_sample,Nt);
X200 = zeros(number_sample,Nt);
dX200 = zeros(number_sample,Nt);

fprintf('Starting simulation: 200-DOF Duffing Nonlinear System...\n');
tic;
parfor j = 1:number_sample;
    ag = Acc(:,j)';
    Y = zeros(2*N, Nt);
    for i = 1:(Nt-1)
        y_n = Y(:, i);

        % Step 1: k1 - slopes at the beginning of the interval
        k1 = duffing_ode(y_n, M_diag, C, k_linear, beta, N, ag(i));

        % Step 2: k2 - slopes at midpoint (using k1 predictor), ag linearly interpolated
        ag_mid = 0.5 * (ag(i) + ag(i+1));
        y2 = y_n + (dt/2) * k1;
        k2 = duffing_ode(y2, M_diag, C, k_linear, beta, N, ag_mid);

        % Step 3: k3 - slopes at midpoint (using k2 predictor)
        y3 = y_n + (dt/2) * k2;
        k3 = duffing_ode(y3, M_diag, C, k_linear, beta, N, ag_mid);

        % Step 4: k4 - slopes at the end of the interval
        y4 = y_n + dt * k3;
        k4 = duffing_ode(y4, M_diag, C, k_linear, beta, N, ag(i+1));

        % Final Update: weighted average of four slopes
        Y(:, i+1) = y_n + (dt/6) * (k1 + 2*k2 + 2*k3 + k4);
    end
    X1(j,:)   = Y(1,   :);
    X200(j,:) = Y(200, :);
    dX200(j,:)= Y(400, :);
end
t_sim1 = toc;
fprintf('Simulation complete! Computation time: %.2f seconds.\n', t_sim1);

%% 4. Post-processing and Visualization
% Calculate restoring force of the 1st floor (Linear + Cubic terms)
F_res_1 = k_linear * X1 + beta * X1.^3;

% Plot 1: Top floor displacement time history
figure(1);
plot(t, X200(1, :), 'Color', [0 0.447 0.741], 'LineWidth', 1);
title(['1st sample of displacement of the ', num2str(N), 'th Floor']);
xlabel('Time (s)'); ylabel('Displacement (m)');
grid on;

% Plot 2: Force-Displacement relationship (Duffing effect)
figure(2);
plot(X1(1,:), F_res_1(1,:), 'Color', [0.85 0.325 0.098], 'LineWidth', 1);
title('1st Floor Restoring Force (Duffing Nonlinearity)');
xlabel('Interstory Drift (m)'); ylabel('Restoring Force (N)');
grid on;

save([Data_path,'X1_response.mat'],'X1', '-v7.3');
save([Data_path,'X200_response.mat'],'X200', '-v7.3');
save([Data_path,'dX200_response.mat'],'dX200', '-v7.3');
save([Data_path,'F_res_1.mat'],'F_res_1', '-v7.3');
save time_response_sim1.mat t_sim1;

end


