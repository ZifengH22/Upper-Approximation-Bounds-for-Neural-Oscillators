function [X,dX,ddX,Z,t_sim] = Y_response(t,dt,M,C,K_in,beta,gamma_r,s,lamda,Acc);
%calculate displacement velocity and acceleration
%M mass matrix;
%C damping matrix;
%K linear stiffness matrix

num_degree = length(M);
C_M = M^-1*C;
K_M = M^-1*K_in;
M_M = M^-1*M;
num_sa = size(Acc,2);
N = size(Acc,1);

%%%%calculation%%%%
X = zeros(N,num_sa);
dX = zeros(N,num_sa);
ddX = zeros(N,num_sa);
Z = zeros(N,num_sa);

X0 = zeros(num_degree,1);
dX0 = zeros(num_degree,1);
Z0 = zeros(num_degree,1); 

tic
parfor ii = 1:num_sa
    F_Ml = ones(num_degree,1)*(Acc(:,ii)');
    [Xout,dXout,ddXout,Zout] = ode_solver(t,dt,M_M,C_M,K_M,beta,gamma_r,s,lamda,X0,dX0,Z0,F_Ml);
    X(:,ii) = Xout(end,:)';
    dX(:,ii) = dXout(end,:)';
    ddX(:,ii) = ddXout(end,:)';
    Z(:,ii) = Zout(end,:)';
end
t_sim = toc;
end





