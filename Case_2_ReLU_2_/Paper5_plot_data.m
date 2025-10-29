clear;clc;close all;
current_path = cd;
Data_path = [current_path,'\data\'];

load([Data_path,'X_train.mat']);
% load([Data_path,'Acc_train.mat']);
% load([Data_path,'X_dX_output_train.mat']);
load([Data_path,'X_pred.mat']);
load([Data_path,'coef_X_output.mat']);

load([Data_path,'Train_epochs_loss.mat']);
load([Data_path,'Val_epochs_loss.mat']);

[num_time,num_sample] = size(X_train);
dt = 0.01;
t = 0:dt:(num_time-1)*dt;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure(1)
semilogy(Train_epochs_loss(:,1),Train_epochs_loss(:,2),'k-',Val_epochs_loss(:,1),Val_epochs_loss(:,2),'b-.');
legend('Training loss (Runge-Kutta neural oscillator)','Validation loss (Runge-Kutta neural oscillator)')
xlim([0,5000]);
ylim([1e-3,1e2]);
xlabel('Epoch');
ylabel('Loss');
set(gca,'fontsize',12);



%%%
error_mse = mean(mean((X_train - X_pred).^2).^0.5)
error_max = max(max(abs(X_train - X_pred)))

%%%
num_sample = length(X_train(1,:));
num_sample_select = 2000;
dnum_sample = num_sample/num_sample_select;
X_train_2000 = zeros(length(X_train(:,1)),num_sample_select);
X_pred_2000 = zeros(length(X_train(:,1)),num_sample_select);
for i = 1:num_sample_select
    X_train_l = X_train(:,(i-1)*dnum_sample+1);
    X_train_2000(:,i) = X_train_l;
    X_pred_l = X_pred(:,(i-1)*dnum_sample+1);
    X_pred_2000(:,i) = X_pred_l;
end

error_mse_2000 = mean(mean((X_train_2000 - X_pred_2000).^2)).^0.5
error_max_2000 = max(max(abs(X_train_2000 - X_pred_2000)))

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



%%%
MSE_oscillator = sum((X_train - X_pred).^2)./sum(X_train.^2);
MSE_network = sum((X_train - X_pred_original).^2)./sum(X_train.^2);

% MSE_oscillator = mean((X_train - X_pred).^2)./mean(X_train(:).^2);
% MSE_network = mean((X_train - X_pred_original).^2)./mean(X_train(:).^2);

xl = linspace(0,0.3,10000);
dxl = mean(diff(xl));

xll = linspace(0,20,10000);
dxll = mean(diff(xll));

alphal = 1000000;
alphall = 10000;
pdf_oscillator = sum(exp(-alphal*(MSE_oscillator'-xl).^2));
pdf_oscillator = pdf_oscillator/sum(pdf_oscillator*dxl);

pdf_network = sum(exp(-alphall*(MSE_network'-xll).^2));
pdf_network = pdf_network/sum(pdf_network*dxll);

figure(2)
plot(xl,pdf_oscillator,'k-')
xlim([0,xl(end)]);
xlabel('Relative MSE');
ylabel('PDF');
set(gca,'fontsize',12);

figure(3)
plot(xll,pdf_network,'b--')
xlim([0,xll(end)]);
xlabel('Relative MSE');
ylabel('PDF');
set(gca,'fontsize',12);

%%%%


bin_number1= 200;
bin_end1 = 20;
bin_edges1 = exp(linspace(log(1e-20),log(bin_end1),bin_number1));

bin_number2= 200;
bin_end2 = 20;
bin_edges2 = exp(linspace(log(1e-20),log(bin_end2),bin_number2));


figure(32)
histogram(MSE_oscillator,bin_edges1, 'FaceColor', 'b', 'FaceAlpha', 1, 'EdgeColor', 'k');
hold on
histogram(MSE_network,bin_edges2, 'FaceColor', 'r', 'FaceAlpha', 1, 'EdgeColor', 'k');
xlim([0.0001,20]);
ylim([0,10000])
set(gca,'xscale','log')
xlabel('Relative MSE');
ylabel('Frequency');
legend('Runge-Kutta neural oscillator','Original Runge-Kutta neural network');
set(gca,'fontsize',12);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[~,indmax] = max(MSE_oscillator)
[~,indmin] = min(MSE_oscillator)

MSE_oscillator(indmax)
MSE_network(indmax)

MSE_oscillator(indmin)
MSE_network(indmin)

figure(4)
plot(t,X_train(:,indmin),'b-',t,X_pred(:,indmin),'k--',t,X_pred_original(:,indmin),'r-.')
xlabel('Time (s)')
ylabel('{\itX}_{10}({\itt})');
ylim([-10,25]);
legend('Simulated sample','Runge-Kutta neural oscillator (MSE: 0.0001)','Original Runge-Kutta neural network (MSE: 0.0673)','Location','southeast');
set(gca,'fontsize',12);

figure(5)
plot(t,X_pred_original(:,indmax),'g.-',t,X_train(:,indmax),'b-',t,X_pred(:,indmax),'k--');
xlabel('Time (s)')
ylabel('{\itX}_{10}({\itt})');
ylim([-10,10]);
legend('Original Runge-Kutta neural network (MSE: 0.885)','Simulated sample','Runge-Kutta neural oscillator (MSE: 0.283)','Location','southeast');
set(gca,'fontsize',12);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
agx = 25;
agy = 20;
fig = figure(6);
for i = 1:5
    location = ones(size(t))*i;
    plot3(location,t,load_sine(i,:));
    hold on
end
xlabel('The number {\iti} of {\itx}_{\iti}({\itt})');
ylabel('Time (s)');
zlabel('{\itx}_{\iti}({\itt})');
zlim([-6,6]);
fig.Position = [100, 100, 560, 420];
view(agx,agy);
set(gca,'fontsize',12);


fig = figure(7);
for i = 6:10
    location = ones(size(t))*i;
    plot3(location,t,load_sine(i,:));
    hold on
end
xlabel('The number {\iti} of {\itx}_{\iti}({\itt})');
ylabel('Time (s)');
zlabel('{\itx}_{\iti}({\itt})');
zlim([-6,6]);
fig.Position = [100, 100, 560, 420];
view(agx,agy);
set(gca,'fontsize',12);

fig = figure(8);
for i = 11:15
    location = ones(size(t))*i;
    plot3(location,t,load_sine(i,:));
    hold on
end
xlabel('The number {\iti} of {\itx}_{\iti}({\itt})');
ylabel('Time (s)');
zlabel('{\itx}_{\iti}({\itt})');
zlim([-6,6]);
fig.Position = [100, 100, 560, 420];
view(agx,agy);
set(gca,'fontsize',12);

fig = figure(9);
for i = 16:20
    location = ones(size(t))*i;
    plot3(location,t,load_sine(i,:));
    hold on
end
xlabel('The number {\iti} of {\itx}_{\iti}({\itt})');
ylabel('Time (s)');
zlabel('{\itx}_{\iti}({\itt})');
zlim([-6,6]);
fig.Position = [100, 100, 560, 420];
view(agx,agy);
set(gca,'fontsize',12);

X_pred_app = 2.8*(load_sine(15,:) + load_sine(16,:))';
MSE_X_pred_app = sum((X_train(:,indmin) - X_pred_app).^2)./sum(X_train(:,indmin).^2)

figure(4)
plot(t,X_pred_original(:,indmin),'g.-',t,X_train(:,indmin),'b-',t,X_pred(:,indmin),'k--',t,X_pred_app,'r-.')
xlabel('Time (s)')
ylabel('{\itX}_{10}({\itt})');
ylim([-15,25]);
legend('Original Runge-Kutta neural network (MSE: 0.0673)','Simulated sample','Runge-Kutta neural oscillator (MSE: 0.0001)','2.8[{\itx}_{15}({\itt})+{\itx}_{16}({\itt})] (MSE: 0.0110)','Location','southeast');
set(gca,'fontsize',12);


% ind = 20
% X_one = X_dX_output_train(ind,:)';
% load_one = Acc_train(:,ind);
% load_one_sine = squeeze(load_sine(:,:,ind))'; 

% fre = 0.04;
% alpha = 0.001;
% phase = 1.3*pi/2;
% load_one_in = load_one';
% load_one_sine_in = load_one_sine(:,1);
% [eps,load_one_sine_est,binom_coeff_inv] = parameter_load_sine(fre,alpha,phase,load_one_in,load_one_sine_in,num_time,dt);
% 
% 
% para0 = [0.01,0.0025,pi/2];
% [paraout,eps] = fminsearch(@(para)parameter_load_sine(para(1),abs(para(2)),para(3),load_one_in,load_one_sine_in,num_time,dt),[1,1,1]);
% [eps,load_one_sine_est,binom_coeff_inv] = parameter_load_sine(paraout(1),abs(paraout(2)),paraout(3),load_one_in,load_one_sine_in,num_time,dt);
% 
% binom_coeff_inv = fliplr(Decay_sin_coef(alpha,fre,phase,dt,num_time))';
% 
% load_sine_est_temp = zeros(size(load_sine_temp));
% for n = 1:num_time
%     % 计算分数阶导数
%     load_templ = load_temp(1:n)';
%     load_sine_est_temp(n,ind) = load_templ*binom_coeff_inv(end-n+1:end); %.*abs(Xout_T)
% end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
t1 = 0.5;
t2 = 5;
t3 = 9;
index1 = t1/dt;
index2 = t2/dt;
index3 = t3/dt;

%%%%
xl = linspace(-6,6,1000);
dxl = mean(diff(xl));
alpha = 10;
pdf1 = sum(exp(-alpha*(X_train(index1,:)'-xl).^2));
pdf1_2000 = sum(exp(-alpha*(X_dX_output_train(1:1600,index1)-xl).^2));
pdf1_oscillator = sum(exp(-alpha*(X_pred(index1,:)'-xl).^2));
pdf1_network = sum(exp(-alpha*(X_pred_original(index1,:)'-xl).^2));

pdf1 = pdf1/sum(pdf1*dxl);
pdf1_2000 = pdf1_2000/sum(pdf1_2000*dxl);
pdf1_oscillator = pdf1_oscillator/sum(pdf1_oscillator*dxl);
pdf1_network = pdf1_network/sum(pdf1_network*dxl);

figure(10)
plot(xl,pdf1_2000,'g.-',xl,pdf1,'b-',xl,pdf1_oscillator,'k--',xl,pdf1_network,'r-.');
xlim([xl(1),xl(end)]);
ylim([0,0.5]);
xlabel('{\itX}_2(0.5)');
ylabel('PDF');
legend('1600 training samples','5e4 simulated samples','Runge-Kutta neural oscillator','Original Runge-Kutta neural network');
set(gca,'fontsize',12);


figure(11)
plot(xl,pdf1_2000,'g.-',xl,pdf1,'b-',xl,pdf1_oscillator,'k--',xl,pdf1_network,'r-.');
xlim([xl(1),xl(end)]);
ylim([1e-20,0.5]);
xlabel('{\itX}_2(0.5)');
ylabel('PDF');
legend('1600 training samples','5e4 simulated samples','Runge-Kutta neural oscillator','Original Runge-Kutta neural network','location','south');
set(gca,'fontsize',12);
set(gca,'yscale','log');

%%%%
xl = linspace(-8,8,1000);
dxl = mean(diff(xl));
alpha = 10;
pdf2 = sum(exp(-alpha*(X_train(index2,:)'-xl).^2));
pdf2_2000 = sum(exp(-alpha*(X_dX_output_train(1:1600,index2)-xl).^2));
pdf2_oscillator = sum(exp(-alpha*(X_pred(index2,:)'-xl).^2));
pdf2_network = sum(exp(-alpha*(X_pred_original(index2,:)'-xl).^2));

pdf2 = pdf2/sum(pdf2*dxl);
pdf2_2000 = pdf2_2000/sum(pdf2_2000*dxl);
pdf2_oscillator = pdf2_oscillator/sum(pdf2_oscillator*dxl);
pdf2_network = pdf2_network/sum(pdf2_network*dxl);

figure(12)
plot(xl,pdf2_2000,'g.-',xl,pdf2,'b-',xl,pdf2_oscillator,'k--',xl,pdf2_network,'r-.');
xlim([xl(1),xl(end)]);
ylim([0,0.25]);
xlabel('{\itX}_2(5)');
ylabel('PDF');
legend('1600 training samples','5e4 simulated samples','Runge-Kutta neural oscillator','Original Runge-Kutta neural network');
set(gca,'fontsize',12);

figure(13)
plot(xl,pdf2_2000,'g.-',xl,pdf2,'b-',xl,pdf2_oscillator,'k--',xl,pdf2_network,'r-.');
xlim([xl(1),xl(end)]);
ylim([1e-10,0.5]);
xlabel('{\itX}_2(5)');
ylabel('PDF');
legend('1600 training samples','5e4 simulated samples','Runge-Kutta neural oscillator','Original Runge-Kutta neural network','location','south');
set(gca,'fontsize',12);
set(gca,'yscale','log');


%%%%
xl = linspace(-6,6,1000);
dxl = mean(diff(xl));
alpha = 10;
pdf3 = sum(exp(-alpha*(X_train(index3,:)'-xl).^2));
pdf3_2000 = sum(exp(-alpha*(X_dX_output_train(1:1600,index3)-xl).^2));
pdf3_oscillator = sum(exp(-alpha*(X_pred(index3,:)'-xl).^2));
pdf3_network = sum(exp(-alpha*(X_pred_original(index3,:)'-xl).^2));

pdf3 = pdf3/sum(pdf3*dxl);
pdf3_2000 = pdf3_2000/sum(pdf3_2000*dxl);
pdf3_oscillator = pdf3_oscillator/sum(pdf3_oscillator*dxl);
pdf3_network = pdf3_network/sum(pdf3_network*dxl);

figure(13)
plot(xl,pdf3_2000,'g.-',xl,pdf3,'b-',xl,pdf3_oscillator,'k--',xl,pdf3_network,'r-.');
xlim([xl(1),xl(end)]);
ylim([0,0.25]);
xlabel('{\itX}_2(9)');
ylabel('PDF');
legend('1600 training samples','5e4 simulated samples','Runge-Kutta neural oscillator','Original Runge-Kutta neural network');
set(gca,'fontsize',12);

figure(14)
plot(xl,pdf3_2000,'g.-',xl,pdf3,'b-',xl,pdf3_oscillator,'k--',xl,pdf3_network,'r-.');
xlim([xl(1),xl(end)]);
ylim([1e-10,0.5]);
xlabel('{\itX}_2(9)');
ylabel('PDF');
legend('1600 training samples','5e4 simulated samples','Runge-Kutta neural oscillator','Original Runge-Kutta neural network','location','south');
set(gca,'fontsize',12);
set(gca,'yscale','log');


%%%%joint PDF of response at two time instants%%%%
t1 = 8;
t2 = 9;
t_index1 = t1/dt;
t_index2 = t2/dt;
x_axis1 = linspace(-6,6,500)';
x_axis2 = linspace(-6,6,500);
dx_axis1 = mean(diff(x_axis1));
dx_axis2 = mean(diff(x_axis2));
h_kernel = 50;


Xflu_sim_l = X_train;
Px = 0;
for i = 1:num_sample
    Px = Px + exp(-h_kernel*(Xflu_sim_l(t_index1,i)-x_axis1).^2)*exp(-h_kernel*(Xflu_sim_l(t_index2,i)-x_axis2).^2);
end
Px = Px/sum(sum(Px*dx_axis1)*dx_axis2);

Xflu_sim_l = X_pred;
Px_oscillator = 0;
for i = 1:num_sample
    Px_oscillator = Px_oscillator + exp(-h_kernel*(Xflu_sim_l(t_index1,i)-x_axis1).^2)*exp(-h_kernel*(Xflu_sim_l(t_index2,i)-x_axis2).^2);
end
Px_oscillator = Px_oscillator/sum(sum(Px_oscillator*dx_axis1)*dx_axis2);


Xflu_sim_l = X_pred_original;
Px_network = 0;
for i = 1:num_sample
    Px_network = Px_network + exp(-h_kernel*(Xflu_sim_l(t_index1,i)-x_axis1).^2)*exp(-h_kernel*(Xflu_sim_l(t_index2,i)-x_axis2).^2);
end
Px_network = Px_network/sum(sum(Px_network*dx_axis1)*dx_axis2);


Xflu_sim_l = X_dX_output_train(1:1600,:)';
Px_1600 = 0;
for i = 1:1600
    Px_1600 = Px_1600 + exp(-h_kernel*(Xflu_sim_l(t_index1,i)-x_axis1).^2)*exp(-h_kernel*(Xflu_sim_l(t_index2,i)-x_axis2).^2);
end
Px_1600 = Px_1600/sum(sum(Px_1600*dx_axis1)*dx_axis2);

agx = -30;
agy = 40;
figure(9)
surf(x_axis1,x_axis2,Px);
xlim([x_axis1(1),x_axis1(end)]);
ylim([x_axis2(1),x_axis2(end)]);
zlim([0,0.08]);
xlabel('{\itX}_2(9)');
ylabel('{\itX}_2(8)');
zlabel('Joint PDF');
set(gca,'fontsize',12);
view(agx,agy);
shading interp;

figure(10)
surf(x_axis1,x_axis2,Px_oscillator);
xlim([x_axis1(1),x_axis1(end)]);
ylim([x_axis2(1),x_axis2(end)]);
zlim([0,0.08]);
xlabel('{\itX}_2(9)');
ylabel('{\itX}_2(8)');
zlabel('Joint PDF');
set(gca,'fontsize',12);
view(agx,agy);
shading interp;

figure(11)
surf(x_axis1,x_axis2,Px_network);
xlim([x_axis1(1),x_axis1(end)]);
ylim([x_axis2(1),x_axis2(end)]);
zlim([0,0.08]);
xlabel('{\itX}_2(9)');
ylabel('{\itX}_2(8)');
zlabel('Joint PDF');
set(gca,'fontsize',12);
view(agx,agy);
shading interp;

%%
figure(12)
surf(x_axis1,x_axis2,Px);
xlim([x_axis1(1),x_axis1(end)]);
ylim([x_axis2(1),x_axis2(end)]);
zlim([0,0.08]);
xlabel('{\itX}_2(9)');
ylabel('{\itX}_2(8)');
zlabel('Joint PDF');
set(gca,'fontsize',12);
view(0,90);
shading interp;

figure(13)
surf(x_axis1,x_axis2,Px_oscillator);
xlim([x_axis1(1),x_axis1(end)]);
ylim([x_axis2(1),x_axis2(end)]);
zlim([0,0.08]);
xlabel('{\itX}_2(9)');
ylabel('{\itX}_2(8)');
zlabel('Joint PDF');
set(gca,'fontsize',12);
view(0,90);
shading interp;

figure(14)
surf(x_axis1,x_axis2,Px_network);
xlim([x_axis1(1),x_axis1(end)]);
ylim([x_axis2(1),x_axis2(end)]);
zlim([0,0.08]);
xlabel('{\itX}_2(9)');
ylabel('{\itX}_2(8)');
zlabel('Joint PDF');
set(gca,'fontsize',12);
view(0,90);
shading interp;

%%%extreme distribution%%%%%
%%%%
xl = linspace(0,8,1000);
dxl = mean(diff(xl));
alpha = 10;
pdf_e = sum(exp(-alpha*(max(abs(X_train))'-xl).^2));
pdf_e_oscillator = sum(exp(-alpha*(max(abs(X_pred))'-xl).^2));
pdf_e_network = sum(exp(-alpha*(max(abs(X_pred_original))'-xl).^2));
pdf_e_1600 = sum(exp(-alpha*(max(abs(X_dX_output_train'))'-xl).^2));


pdf_e = pdf_e/sum(pdf_e*dxl);
pdf_e_1600 = pdf_e_1600/sum(pdf_e_1600*dxl);
pdf_e_oscillator = pdf_e_oscillator/sum(pdf_e_oscillator*dxl);
pdf_e_network = pdf_e_network/sum(pdf_e_network*dxl);

figure(14)
plot(xl,pdf_e,'b-',xl,pdf_e_oscillator,'k--',xl,pdf_e_network,'r-.');
xlim([xl(1),xl(end)]);
ylim([0,1.2]);
xlabel('max[|{\itX}_2({\itt})|]');
ylabel('PDF');
legend('5e4 simulated samples','Runge-Kutta neural oscillator','Original Runge-Kutta neural network');
set(gca,'fontsize',12);


figure(15)
plot(xl,pdf_e_1600,'g.-',xl,pdf_e,'b-',xl,pdf_e_oscillator,'k--',xl,pdf_e_network,'r-.');
xlim([xl(1),xl(end)]);
ylim([0,1.2]);
xlabel('max[|{\itX}_2({\itt})|]');
ylabel('PDF');
legend('1600 training samples','5e4 simulated samples','Runge-Kutta neural oscillator','Original Runge-Kutta neural network');
set(gca,'fontsize',12);
set(gca,'yscale','log');