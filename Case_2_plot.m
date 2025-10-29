clear;clc;close all;

%%%%%%%%%%%loading data%%%%%%%%%%%
current_path = cd;
dir_info = dir(current_path);
dir_info = dir_info([dir_info.isdir]); % Keep only directories
dir_info = dir_info(~ismember({dir_info.name}, {'.', '..'})); % Exclude 
dir_names = {dir_info.name};

file_names_suffix = {'Case_2_ReLU_2_';'Case_2_ReLU_5_';'Case_2_ReLU_10_';'Case_2_ReLU_20_';'Case_2_ReLU_30_';'Case_2_ReLU_40_'};
[number_filefolder,~] = size(file_names_suffix);

for i = 1:number_filefolder
    file_name_suffix_l = cellstr(file_names_suffix{i});
    matchingIdx = find(contains(dir_names, file_name_suffix_l) == 1);
    file_name_l = [current_path,'\',dir_names{matchingIdx}];
    file_name_l_data = [file_name_l,'\data'];
    file_name_l_data_X_train = [file_name_l_data,'\X_train.mat'];
    file_name_l_data_X_pred = [file_name_l_data,'\X_pred.mat'];
    file_name_l_data_Train_epochs_loss = [file_name_l_data,'\Train_epochs_loss.mat'];
    file_name_l_data_Val_epochs_loss = [file_name_l_data,'\Val_epochs_loss.mat'];

    assignin('base', ['X_train_',file_names_suffix{i}([6:end])], load(file_name_l_data_X_train).X_train);
    assignin('base', ['X_pred_',file_names_suffix{i}([6:end])], load(file_name_l_data_X_pred).X_pred);
    assignin('base', ['Train_epochs_loss_',file_names_suffix{i}([6:end])], load(file_name_l_data_Train_epochs_loss).Train_epochs_loss);
    assignin('base', ['Val_epochs_loss_',file_names_suffix{i}([6:end])], load(file_name_l_data_Val_epochs_loss).Val_epochs_loss);
end

%%%%%%%%%%%calculation and plot%%%%%%%%%%%
width = [2,5,10,20,30,40];
error_mse_data = zeros(size(width));
error_max_data = zeros(size(width));
for i = 1:number_filefolder
    % commandstr_error_mse_data = ['error_mse_data(i) = ','mean(mean((',['X_train_',file_names_suffix{i}([6:end])], '-', ['X_pred_',file_names_suffix{i}([6:end])],').^2)).^0.5;'];
    commandstr_error_mse_data = ['error_mse_data(i) = ','mean(mean((',['X_train_',file_names_suffix{i}([6:end])], '-', ['X_pred_',file_names_suffix{i}([6:end])],').^2)).^0.5/','mean(mean((',['X_train_',file_names_suffix{i}([6:end])],').^2)).^0.5',';'];
    eval(commandstr_error_mse_data);

    % commandstr_error_max_data = ['error_max_data(i) = ','max(max(abs(',['X_train_',file_names_suffix{i}([6:end])], '-', ['X_pred_',file_names_suffix{i}([6:end])],')));'];
    commandstr_error_max_data = ['error_max_data(i) = ','max(max(abs(',['X_train_',file_names_suffix{i}([6:end])], '-', ['X_pred_',file_names_suffix{i}([6:end])],')))/','max(max(abs(',['X_train_',file_names_suffix{i}([6:end])],')));'];
    eval(commandstr_error_max_data);
end

a = 0;
b = 1.2;
x_width = linspace(width(1),width(end),50);
error_max_ana = a+b*x_width.^(-0.5);
figure(1)
plot(width,error_max_data,'*',x_width,error_max_ana);
% xlabel('The hidden layer width of $\it\Gamma(\cdot)$', 'Interpreter', 'latex');
% ylabel('Relative error under {\it L}-$\infty$ norm', 'Interpreter', 'latex');
xlabel('$w_{\it\Gamma_i}$', 'Interpreter', 'latex');
ylabel('$\tilde{\varepsilon}_{X,\infty}$', 'Interpreter', 'latex');
legend('Numerical results', ['$\tilde{\varepsilon}_{X,\infty} = ',num2str(b),'w_{\it\Gamma_i}^{-0.5}$'], 'Interpreter', 'latex');
set(gca,'fontsize',15);
% print('Fig 2(a) Relative_error_under_supremum_norm', '-depsc','-r300');
exportgraphics(gcf,'Fig 2(a) Relative_error_under_supremum_norm.pdf','Resolution',300);
savefig('Fig 2(a) Relative_error_under_supremum_norm.fig');

a = 0;
b = 1;
x_width = linspace(width(1),width(end),50);
error_mse_ana = a+b*x_width.^(-0.5);
figure(2)
plot(width,error_mse_data,'*',x_width,error_mse_ana);
% xlabel('The hidden layer width of $\it\Gamma(\cdot)$', 'Interpreter', 'latex');
% ylabel('Relative error under {\it L}-2 norm', 'Interpreter', 'latex');
xlabel('$w_{\it\Gamma_i}$', 'Interpreter', 'latex');
ylabel('$\tilde{\varepsilon}_{X,2}$', 'Interpreter', 'latex');
legend('Numerical results', ['$\tilde{\varepsilon}_{X,2} = ','w_{\it\Gamma_i}^{-0.5}$'], 'Interpreter', 'latex');
set(gca,'fontsize',15);
% print('Fig 2(b) Relative_error_under_L2_norm', '-depsc','-r300');
exportgraphics(gcf,'Fig 2(b) Relative_error_under_L2_norm.pdf','Resolution',300);
savefig('Fig 2(b) Relative_error_under_L2_norm.fig');


%%%%%%%%%%%convergence error%%%%%%%%%%%%%%%%%%
figure(3)
plot(Train_epochs_loss_2_ReLU_2_(:,1),Train_epochs_loss_2_ReLU_2_(:,2),'b-',Val_epochs_loss_2_ReLU_2_(:,1),Val_epochs_loss_2_ReLU_2_(:,2),'r-.');
hold on;
plot(Train_epochs_loss_2_ReLU_5_(:,1),Train_epochs_loss_2_ReLU_5_(:,2),'b-',Val_epochs_loss_2_ReLU_5_(:,1),Val_epochs_loss_2_ReLU_5_(:,2),'r-.');
plot(Train_epochs_loss_2_ReLU_10_(:,1),Train_epochs_loss_2_ReLU_10_(:,2),'b-',Val_epochs_loss_2_ReLU_10_(:,1),Val_epochs_loss_2_ReLU_10_(:,2),'r-.');
plot(Train_epochs_loss_2_ReLU_20_(:,1),Train_epochs_loss_2_ReLU_20_(:,2),'b-',Val_epochs_loss_2_ReLU_20_(:,1),Val_epochs_loss_2_ReLU_20_(:,2),'r-.');
plot(Train_epochs_loss_2_ReLU_30_(:,1),Train_epochs_loss_2_ReLU_30_(:,2),'b-',Val_epochs_loss_2_ReLU_30_(:,1),Val_epochs_loss_2_ReLU_30_(:,2),'r-.');
plot(Train_epochs_loss_2_ReLU_40_(:,1),Train_epochs_loss_2_ReLU_40_(:,2),'b-',Val_epochs_loss_2_ReLU_40_(:,1),Val_epochs_loss_2_ReLU_40_(:,2),'r-.');
legend('Training loss','Validation loss', 'Interpreter', 'latex');
xlim([0,5000]);
ylim([1e-3,1e2]);
xlabel('Epoch', 'Interpreter', 'latex');
ylabel(['$\ell_{', num2str(2), '}$'], 'Interpreter', 'latex');
set(gca,'fontsize',15);
set(gca,'XScale','linear','YScale','log');