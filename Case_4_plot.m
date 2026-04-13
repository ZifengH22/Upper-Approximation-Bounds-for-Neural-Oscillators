clear;clc;close all;

%%%%%%%%%%%loading data%%%%%%%%%%%
current_path = cd;
dir_info = dir(current_path);
dir_info = dir_info([dir_info.isdir]); % Keep only directories
dir_info = dir_info(~ismember({dir_info.name}, {'.', '..'})); % Exclude 
dir_names = {dir_info.name};

% file_names_suffix = {'Case_4_10width'};
file_names_suffix = {'Case_4_'};

file_name_suffix_l = cellstr(file_names_suffix);
matchingIdx = find(contains(dir_names, file_name_suffix_l) == 1);
file_name_l = [current_path,'\',dir_names{matchingIdx(1)}];
file_name_l_data = [file_name_l,'\data'];
file_name_l_data_X_train = [file_name_l_data,'\X200_response.mat'];
assignin('base', ['X200'], load(file_name_l_data_X_train).X200);

% width_vector = [2,5,10,20,30,40,60,80];
width_vector = [2,5,10,20,30,40,60];
number_width = length(width_vector);
for i = 1:number_width
    % Build file names
    file_name_l_data_X_pred = [file_name_l_data,'\X_pred_',num2str(width_vector(i)),'.mat'];
    file_name_l_data_Train_epochs_loss = [file_name_l_data,'\Train_epochs_loss_',num2str(width_vector(i)),'.mat'];
    file_name_l_data_Val_epochs_loss = [file_name_l_data,'\Val_epochs_loss_',num2str(width_vector(i)),'.mat'];

    % Load .mat files
    temp_X = load(file_name_l_data_X_pred);
    temp_Train = load(file_name_l_data_Train_epochs_loss);
    temp_Val = load(file_name_l_data_Val_epochs_loss);

    % Assign variables to base workspace using dynamic field names
    assignin('base', ['X_pred_',num2str(width_vector(i))], temp_X.(['X_pred_',num2str(width_vector(i))]));
    assignin('base', ['Train_epochs_loss_',num2str(width_vector(i))], temp_Train.(['Train_epochs_loss_',num2str(width_vector(i))]));
    assignin('base', ['Val_epochs_loss_',num2str(width_vector(i))], temp_Val.(['Val_epochs_loss_',num2str(width_vector(i))]));
end

%%%%%%%%%%%calculation and plot%%%%%%%%%%%

error_mse_data = zeros(size(width_vector));
error_max_data = zeros(size(width_vector));
for i = 1:number_width
    commandstr_error_mse_data = ['error_mse_data(i) = ','mean(mean((X200', '-', 'X_pred_',num2str(width_vector(i)),').^2)).^0.5/','mean(mean(X200.^2)).^0.5;'];
    eval(commandstr_error_mse_data);

    commandstr_error_max_data = ['error_max_data(i) = ','max(max(abs(X200', '-', 'X_pred_',num2str(width_vector(i)),')))/','max(max(abs(X200)));'];
    eval(commandstr_error_max_data);
end

a1 = 0;
b1 = 0.6;
c1 = -0.5;
x_width = linspace(width_vector(1),width_vector(end),50);
error_max_ana = a1+b1*x_width.^(c1);
figure(1)
plot(width_vector,error_max_data,'*',x_width,error_max_ana);
% xlabel('$w_{{\it\Gamma}_i}$', 'Interpreter', 'latex');
xlabel('$w_{i}$', 'Interpreter', 'latex');
ylabel('$\tilde{\varepsilon}_{X_{200},\infty}$', 'Interpreter', 'latex');
% legend('Numerical results', ['$\tilde{\varepsilon}_{X_{200},\infty} = ',num2str(b1),'w_{{\it\Gamma}_i}^{',num2str(c1),'}$'], 'Interpreter', 'latex');
legend('Numerical results', ['$\tilde{\varepsilon}_{X_{200},\infty} = ',num2str(b1),'w_{i}^{',num2str(c1),'}$'], 'Interpreter', 'latex');
set(gca,'fontsize',15);
exportgraphics(gcf,'Fig_5_a_Relative_error_under_supremum_norm.pdf','Resolution',300);
savefig('Fig_5_a_Relative_error_under_supremum_norm.fig');

a2 = 0;
b2 = 0.4;
c2 = -0.5;
x_width = linspace(width_vector(1),width_vector(end),50);
error_mse_ana = a2+b2*x_width.^(c2);
figure(2)
plot(width_vector,error_mse_data,'*',x_width,error_mse_ana);
% xlabel('$w_{{\it\Gamma}_i}$', 'Interpreter', 'latex');
xlabel('$w_{i}$', 'Interpreter', 'latex');
ylabel('$\tilde{\varepsilon}_{X_{200},2}$', 'Interpreter', 'latex');
% legend('Numerical results', ['$\tilde{\varepsilon}_{X_{200},2} = ',num2str(b2),'w_{{\it\Gamma}_i}^{',num2str(c2),'}$'], 'Interpreter', 'latex');
legend('Numerical results', ['$\tilde{\varepsilon}_{X_{200},2} = ',num2str(b2),'w_{i}^{',num2str(c2),'}$'], 'Interpreter', 'latex');

set(gca,'fontsize',15);
exportgraphics(gcf,'Fig_5_b_Relative_error_under_L2_norm.pdf','Resolution',300);
savefig('Fig_5_b_Relative_error_under_L2_norm.fig');

%%%%%%%%%convergence error%%%%%%%%%%%%%%%%%%
figure(3)
plot(Train_epochs_loss_2(:,1),Train_epochs_loss_2(:,2),'b-',Val_epochs_loss_2(:,1),Val_epochs_loss_2(:,2),'r-.');
hold on;
plot(Train_epochs_loss_5(:,1),Train_epochs_loss_5(:,2),'b-',Val_epochs_loss_5(:,1),Val_epochs_loss_5(:,2),'r-.');
plot(Train_epochs_loss_10(:,1),Train_epochs_loss_10(:,2),'b-',Val_epochs_loss_10(:,1),Val_epochs_loss_10(:,2),'r-.');
plot(Train_epochs_loss_20(:,1),Train_epochs_loss_20(:,2),'b-',Val_epochs_loss_20(:,1),Val_epochs_loss_20(:,2),'r-.');
plot(Train_epochs_loss_30(:,1),Train_epochs_loss_30(:,2),'b-',Val_epochs_loss_30(:,1),Val_epochs_loss_30(:,2),'r-.');
plot(Train_epochs_loss_40(:,1),Train_epochs_loss_40(:,2),'b-',Val_epochs_loss_40(:,1),Val_epochs_loss_40(:,2),'r-.');
plot(Train_epochs_loss_60(:,1),Train_epochs_loss_60(:,2),'b-',Val_epochs_loss_60(:,1),Val_epochs_loss_60(:,2),'r-.');
% plot(Train_epochs_loss_80(:,1),Train_epochs_loss_80(:,2),'b-',Val_epochs_loss_80(:,1),Val_epochs_loss_80(:,2),'r-.');
legend('Training loss','Validation loss', 'Interpreter', 'latex');
% xlim([0,5000]);
% ylim([1e-3,1e2]);
xlabel('Epoch', 'Interpreter', 'latex');
ylabel(['$\ell_{', num2str(2), '}$'], 'Interpreter', 'latex');
set(gca,'fontsize',15);
set(gca,'XScale','line','YScale','log');



%%%%%%%%%PDF and CDF of EX%%%%%%%%%%%%
E_X_range = linspace(1,150,100);

E_X_pred = max(abs(X_pred_30)');
E_X200 = max(abs(X200)');

[PDF_E_X_pred,~] = ksdensity(E_X_pred(end,:),E_X_range,'Bandwidth',1);
[PDF_E_X200,~] = ksdensity(E_X200(end,:),E_X_range,'Bandwidth',1);

figure(4)
plot(E_X_range,PDF_E_X_pred,'k-',E_X_range,PDF_E_X200,'b--');
xlim([0,160]);
legend('$\it{X_{\mathrm{200}}}$','$\it{\tilde{X}_{\mathrm{200}}}$', 'Interpreter', 'latex');
xlabel('${\it{E}}_{\mathrm{max}}$', 'Interpreter', 'latex');
ylabel('PDF', 'Interpreter', 'latex');
set(gca,'fontsize',15);
set(gcf, 'Position', [100 100 550 400]);           
set(gcf, 'PaperPosition', [0 0 5.5 4]);            
set(gcf, 'PaperSize', [5.5 4]);
set(gca,'yscale','linear');
exportgraphics(gcf,'Fig_6_a_PDF_EX.pdf','Resolution',300);
savefig('Fig_6_a_PDF_EX.fig');


[CDF_E_X_pred,E_X_pred_range] = ecdf(E_X_pred);
[CDF_E_X200,E_X200_range] = ecdf(E_X200);
CDF_E_X200_y = CDF_E_X200;

eps = 1e-5;
CDF_E_X_pred = min(max(CDF_E_X_pred, eps), 1 - eps);
CDF_E_X200 = min(max(CDF_E_X200, eps), 1 - eps);
CDF_E_X200_y = min(max(CDF_E_X200_y, eps/2), 1 - eps/2);

z_pred = norminv(CDF_E_X_pred);
z_1 = norminv(CDF_E_X200);
z_1_y = norminv(CDF_E_X200_y);
z_1_y_5 = linspace(z_1_y(1),z_1_y(end),5);
idx = zeros(size(z_1_y_5)); 
for i = 1:length(z_1_y_5)
    [~, idx(i)] = min(abs(z_1_y - z_1_y_5(i)));
end
prob_labels = CDF_E_X200_y(idx);

figure(5)
plot(E_X_pred_range,z_pred,'bo',E_X200_range,z_1,'r*');
xlim([0,160]);
legend('$\it{X_{\mathrm{200}}}$','$\it{\tilde{X}_{\mathrm{200}}}$', 'Interpreter', 'latex','Location','northwest');
xlabel('${\it{E}}_{\mathrm{max}}$', 'Interpreter', 'latex');
ylabel('CDF', 'Interpreter', 'latex');
yticks(z_1_y_5);
yticklabels(compose('%.3f', prob_labels));
ylim([min(z_1_y), max(z_1_y)]);
set(gca,'fontsize',15);
set(gcf, 'Position', [100 100 550 400]);           
set(gcf, 'PaperPosition', [0 0 5.5 4]);            
set(gcf, 'PaperSize', [5.5 4]);
set(gca,'xscale','linear','yscale','linear');
exportgraphics(gcf,'Fig_6_b_CDF_EX.pdf','Resolution',300);
savefig('Fig_6_b_CDF_EX.fig');
