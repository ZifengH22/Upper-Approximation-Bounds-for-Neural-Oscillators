clear;clc;close all;

%%%%%%%%%%%loading data%%%%%%%%%%%
current_path = cd;
dir_info = dir(current_path);
dir_info = dir_info([dir_info.isdir]); % Keep only directories
dir_info = dir_info(~ismember({dir_info.name}, {'.', '..'})); % Exclude 
dir_names = {dir_info.name};

file_names_suffix = {'Case_1_'};
file_name_suffix_l = cellstr(file_names_suffix);
matchingIdx = find(contains(dir_names, file_name_suffix_l) == 1);
file_name_l = [current_path,'\',dir_names{matchingIdx}];
file_name_l_data = [file_name_l,'\data'];
file_name_l_data_X_train = [file_name_l_data,'\E_X_train.mat'];
assignin('base', ['E_X_train'], load(file_name_l_data_X_train).E_X_train);

% number_depth = 5;
% for i = 1:number_depth
%     file_name_l_data_X_pred = [file_name_l_data,'\E_X_pred_',num2str(i),'.mat'];
%     file_name_l_data_Train_epochs_loss = [file_name_l_data,'\Train_epochs_loss_',num2str(i),'.mat'];
%     file_name_l_data_Val_epochs_loss = [file_name_l_data,'\Val_epochs_loss_',num2str(i),'.mat'];
%     assignin('base', ['E_X_pred_',num2str(i)], load(file_name_l_data_X_pred).['E_X_pred_',num2str(i)]);
%     assignin('base', ['Train_epochs_loss_',num2str(i)], load(file_name_l_data_Train_epochs_loss));
%     assignin('base', ['Val_epochs_loss_',num2str(i)], load(file_name_l_data_Val_epochs_loss));
% end

number_depth = 5;
for i = 1:number_depth
    % Build file names
    file_name_l_data_X_pred = [file_name_l_data,'\E_X_pred_',num2str(i),'.mat'];
    file_name_l_data_Train_epochs_loss = [file_name_l_data,'\Train_epochs_loss_',num2str(i),'.mat'];
    file_name_l_data_Val_epochs_loss = [file_name_l_data,'\Val_epochs_loss_',num2str(i),'.mat'];

    % Load .mat files
    temp_X = load(file_name_l_data_X_pred);
    temp_Train = load(file_name_l_data_Train_epochs_loss);
    temp_Val = load(file_name_l_data_Val_epochs_loss);

    % Assign variables to base workspace using dynamic field names
    assignin('base', ['E_X_pred_',num2str(i)], temp_X.(['E_X_pred_',num2str(i)]));
    assignin('base', ['Train_epochs_loss_',num2str(i)], temp_Train.(['Train_epochs_loss_',num2str(i)]));
    assignin('base', ['Val_epochs_loss_',num2str(i)], temp_Val.(['Val_epochs_loss_',num2str(i)]));
end

%%%%%%%%%%%calculation and plot%%%%%%%%%%%
depth= [1,2,3,4,5];
error_mse_data = zeros(size(depth));
error_max_data = zeros(size(depth));
for i = 1:number_depth
    commandstr_error_mse_data = ['error_mse_data(i) = ','mean(mean((E_X_train', '-', 'E_X_pred_',num2str(i),').^2)).^0.5/','mean(mean(E_X_train.^2)).^0.5;'];
    eval(commandstr_error_mse_data);

    commandstr_error_max_data = ['error_max_data(i) = ','max(max(abs(E_X_train', '-', 'E_X_pred_',num2str(i),')))/','max(max(abs(E_X_train)));'];
    eval(commandstr_error_max_data);
end

a = 0;
b = 0.088;
x_depth = linspace(depth(1),depth(end),50);
error_max_ana = a+b*x_depth.^(-1/7);
figure(1)
plot(depth,error_max_data,'*',x_depth,error_max_ana);
ylim([0.065,0.095])
xlabel('$H_{\it\Pi_i}$', 'Interpreter', 'latex');
ylabel('$\tilde{\varepsilon}_{E_X,\infty}$', 'Interpreter', 'latex');
legend('Numerical results', ['$\tilde{\varepsilon}_{E_X,\infty} = ',num2str(b),'H_{\it\Pi_i}^{-1/7}$'], 'Interpreter', 'latex');
set(gca,'fontsize',15);
% print('Fig 1(a) Relative_error_under_supremum_norm', '-depsc','-r300');
exportgraphics(gcf,'Fig 1(a) Relative_error_under_supremum_norm.pdf','Resolution',300);
savefig('Fig 1(a) Relative_error_under_supremum_norm.fig');

a = 0;
b = 0.066;
x_depth = linspace(depth(1),depth(end),50);
error_mse_ana = a+b*x_depth.^(-1/7);
figure(2)
plot(depth,error_mse_data,'*',x_depth,error_mse_ana);
xlabel('$H_{\it\Pi_i}$', 'Interpreter', 'latex');
ylabel('$\tilde{\varepsilon}_{E_X,2}$', 'Interpreter', 'latex');
legend('Numerical results', ['$\tilde{\varepsilon}_{E_X,2} = ',num2str(b),'H_{\it\Pi_i}^{-1/7}$'], 'Interpreter', 'latex');
set(gca,'fontsize',15);
% print('Fig 1(b) Relative_error_under_L2_norm', '-depsc','-r300');
exportgraphics(gcf,'Fig 1(b) Relative_error_under_L2_norm.pdf','Resolution',300);
savefig('Fig 1(b) Relative_error_under_L2_norm.fig');

%%%%%%%%%%%convergence error%%%%%%%%%%%%%%%%%%
figure(3)
plot(Train_epochs_loss_1(:,1),Train_epochs_loss_1(:,2),'b-',Val_epochs_loss_1(:,1),Val_epochs_loss_1(:,2),'r-.');
hold on;
plot(Train_epochs_loss_2(:,1),Train_epochs_loss_2(:,2),'b-',Val_epochs_loss_2(:,1),Val_epochs_loss_2(:,2),'r-.');
plot(Train_epochs_loss_3(:,1),Train_epochs_loss_3(:,2),'b-',Val_epochs_loss_3(:,1),Val_epochs_loss_3(:,2),'r-.');
plot(Train_epochs_loss_4(:,1),Train_epochs_loss_4(:,2),'b-',Val_epochs_loss_4(:,1),Val_epochs_loss_4(:,2),'r-.');
plot(Train_epochs_loss_5(:,1),Train_epochs_loss_5(:,2),'b-',Val_epochs_loss_5(:,1),Val_epochs_loss_5(:,2),'r-.');
legend('Training loss','Validation loss', 'Interpreter', 'latex');
% xlim([0,5000]);
% ylim([1e-3,1e2]);
xlabel('Epoch', 'Interpreter', 'latex');
ylabel(['$\ell_{', num2str(8), '}$'], 'Interpreter', 'latex');
set(gca,'fontsize',15);
set(gca,'XScale','linear','YScale','log');