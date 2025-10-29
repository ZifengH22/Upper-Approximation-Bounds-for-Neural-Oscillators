function [Xout,Vout,Aout,Zout] = ode_solver(t_in,dt,Min,Cin,Kin,beta,gamma_r,s,lamda,X0,V0,Z0,F_in)
% 参数设置
% 
% t_in = t;
% F_in = F_Ml*200;
% Min = M_M;
% Cin = C_M;
% Kin = K_M;
% V0 = dX0;

num_degree = length(X0);
Trans = eye(num_degree) - diag(diag(eye(num_degree-1)), -1);
N = length(t_in);      % 时间步数

% 初始条件
Xout = zeros(num_degree, N);    % 位移
Vout = zeros(num_degree, N);    % 速度
Zout = zeros(num_degree, N);    % 位移
Aout = zeros(num_degree, N);    % 加速度

% 初始位移和速度
Xout(:, 1) = X0;
Vout(:, 1) = V0;
Zout(:, 1) = Z0;
Aout(:, 1) = F_in(:, 1) - Cin*Vout(:, 1) - lamda*Kin*Trans*Xout(:, 1) - (1 - lamda)*Kin*Zout(:, 1);
 

% 数值求解
for n = 1:N-1

    %%%k0
    Xl = Xout(:, n);
    Vl = Vout(:, n);
    Zl = Zout(:, n);
    Fl = F_in(:, n);

    k01 = Vl;
    k02 = (Fl - Cin*Vl - lamda*Kin*Trans*Xl - (1 - lamda)*Kin*Zl);
    k03 = Trans*Vl - beta*abs(Trans*Vl).*abs(Zl).^(s - 1).*Zl - gamma_r*(Trans*Vl).*abs(Zl).^s;
    
    %%%k1
    Xl = Xl + 0.5*dt*k01;
    Vl = Vl + 0.5*dt*k02;
    Zl = Zl + 0.5*dt*k03;
    Fl = 0.5*(F_in(:, n) + F_in(:, n + 1));

    k11 = Vl;
    k12 = (Fl - Cin*Vl - lamda*Kin*Trans*Xl - (1 - lamda)*Kin*Zl);
    k13 = Trans*Vl - beta*abs(Trans*Vl).*abs(Zl).^(s - 1).*Zl - gamma_r*(Trans*Vl).*abs(Zl).^s;
    
    %%%k2
    Xl = Xl + 0.5*dt*k11;
    Vl = Vl + 0.5*dt*k12;
    Zl = Zl + 0.5*dt*k13;
    %Fl = 0.5*(F_in(:, n) + F_in(:, n + 1));
    
    k21 = Vl;
    k22 = (Fl - Cin*Vl - lamda*Kin*Trans*Xl - (1 - lamda)*Kin*Zl);
    k23 = Trans*Vl - beta*abs(Trans*Vl).*abs(Zl).^(s - 1).*Zl - gamma_r*(Trans*Vl).*abs(Zl).^s;
    
    %%%k3
    Xl = Xl + dt*k21;
    Vl = Vl + dt*k22;
    Zl = Zl + dt*k23;
    Fl = F_in(:, n + 1);
    
    k31 = Vl;
    k32 = (Fl - Cin*Vl - lamda*Kin*Trans*Xl - (1 - lamda)*Kin*Zl);
    k33 = Trans*Vl - beta*abs(Trans*Vl).*abs(Zl).^(s - 1).*Zl - gamma_r*(Trans*Vl).*abs(Zl).^s;

    %%% final
    Xout(:, n+1) = Xout(:, n) + 1/6*dt*(k01 + 2*k11 + 2*k21 + k31);
    Vout(:, n+1) = Vout(:, n) + 1/6*dt*(k02 + 2*k12 + 2*k22 + k32);
    Zout(:, n+1) = Zout(:, n) + 1/6*dt*(k03 + 2*k13 + 2*k23 + k33);
    Aout(:, n+1) = F_in(:, n+1) - Cin*Vout(:, n+1) - lamda*Kin*Trans*Xout(:, n+1) - (1 - lamda)*Kin*Zout(:, n+1);
end


end