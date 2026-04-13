clear;clc;close all;

%%%%%%%%%%%loading data%%%%%%%%%%%
current_path = cd;
target_path = [current_path,'\Case_3_\'];
cd(target_path)
load([target_path,'data\Acc.mat']);

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
Acc = Acc';
Acc_lr = fliplr(Acc);
Acc_max = max(max(abs(Acc)));
Acc_std = mean(mean(Acc.^2)).^0.5;
LK = max(max(abs(diff(Acc')/dt)));

%%%%%%%%%%%%%%%%%
c_K = 1.1;
M_limit = (10*c_K);
M_limit = max(M_limit,1);
M_number = 10;
M = floor(M_limit*(2.^[0:1:(M_number-1)]'));
M(M>num_time/2) = 5000;
r = floor(log(M));
v = T*r.^(1./(r+1)).*(c_K).*(factorial(r).^(2./(r+1))).*(log(M).^(1./(r+1)))./(M.^(r./(r+1)));

%%%%%%%%%%%%%%%%%
Eps_sup = zeros(1,M_number);
Eps_L2 = zeros(1,M_number);

for i = 1:M_number
    rou_v = bump_rou_v(t_total,v(i),dt);
    fre_temp = [1:M(i)]*df;
    exp_mat_temp = exp(-2*pi*sqrt(-1)*t_total'*fre_temp);

    F_rou_v = rou_v*exp_mat_temp*dt;

    F_Acc_zero_initial_lr = -sqrt(-1)*(-Acc_lr*imag(exp_mat_temp(t_total_length/2:1:(t_total_length/2+t_total_length/4-1),:))*dt + Acc(:,1)./(2*pi*fre_temp).*(cos(2*pi*fre_temp*T)-1)  );
    Acc_zero_initial_lr_estimate = real((F_Acc_zero_initial_lr.*F_rou_v)*exp_mat_temp(t_total_length/2:1:(t_total_length/2+t_total_length/4-1),:)'/T);
    Acc_lr_estimate = Acc_zero_initial_lr_estimate + Acc(:,1); 
    Eps_sup(i) = max(max(abs(Acc_lr_estimate - Acc_lr)))/Acc_max;
    Eps_L2(i) = mean(mean((Acc_lr_estimate - Acc_lr).^2)).^0.5/Acc_std;
end

%%%%%%%%%plot result%%%%%%%%%
cd(current_path)

a1 = 0;
b1 = 7;
c1 = -1;
M_width = linspace(M(1),M(end),500);
error_max_ana = a1 + b1*log(M_width).^2.*M_width.^c1; 

figure(1)
loglog(M,Eps_sup,'*',M_width,error_max_ana);
xlim([M(1),M(end)]);
ylim([0.07,4]);
xlabel('$M_{\it\Gamma}$', 'Interpreter', 'latex');
ylabel('$\tilde{\varepsilon}_{U,\infty}$', 'Interpreter', 'latex');
legend('Numerical results', ['$\tilde{\varepsilon}_{U,\infty} = ',num2str(b1),'(\ln{M}_{\it{\Gamma}})^2/\it{M_{\Gamma}}$'], 'Interpreter', 'latex');
set(gca,'fontsize',15);
exportgraphics(gcf,'Fig_4_a_Relative_error_under_supremum_norm.pdf','Resolution',300);
savefig('Fig_4_a_Relative_error_under_supremum_norm.fig');


a2 = 0;
b2 = 7;
c2 = -1;
M_width = linspace(M(1),M(end),500);
error_mse_ana = a2 + b2*log(M_width).^2.*M_width.^c2; 

figure(2)
loglog(M,Eps_L2,'*',M_width,error_mse_ana);
xlim([M(1),M(end)]);
ylim([0.07,4]);
xlabel('$M_{\it\Gamma}$', 'Interpreter', 'latex');
ylabel('$\tilde{\varepsilon}_{U,2}$', 'Interpreter', 'latex');
legend('Numerical results', ['$\tilde{\varepsilon}_{U,2} = ',num2str(b2),'(\ln{M}_{\it{\Gamma}})^2/\it{M_{\Gamma}}$'], 'Interpreter', 'latex');
set(gca,'fontsize',15);
exportgraphics(gcf,'Fig_4_b_Relative_error_under_L2_norm.pdf','Resolution',300);
savefig('Fig_4_b_Relative_error_under_L2_norm.fig');