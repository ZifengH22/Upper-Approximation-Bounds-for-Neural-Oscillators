loss_L = 8;
depth = [1,2,3,4,5];
error_mse_data = [0.066464,0.060161,0.057182,0.053119,0.051553];
error_max_data = [0.089752,0.078687,0.077643,0.069601,0.067618];
error_loss_data = [0.1349,0.12448,0.11844,0.11043,0.10729];

a = 0;
b = 0.088;
x_depth = linspace(depth(1),depth(end),50);
error_max_ana = a+b*x_depth.^(-1/7);
figure(61)
plot(depth,error_max_data,'*',x_depth,error_max_ana);
ylim([0.065,0.095])
xlabel('$H_{\it\Pi}$', 'Interpreter', 'latex');
ylabel('$\tilde{\varepsilon}_{E_X,\infty}$', 'Interpreter', 'latex');
legend('Numerical results', ['$\tilde{\varepsilon}_{E_X,\infty} = ',num2str(b),'H_{\it\Pi}^{-1/7}$'], 'Interpreter', 'latex');
set(gca,'fontsize',15);
% print('Fig 1(a) Relative_error_under_supremum_norm', '-depsc','-r300');
% exportgraphics(gcf,'Fig 1(a) Relative_error_under_supremum_norm.pdf','Resolution',300);
% savefig('Fig 1(a) Relative_error_under_supremum_norm.fig');

a = 0;
b = 0.0665;
x_depth = linspace(depth(1),depth(end),50);
error_mse_ana = a+b*x_depth.^(-1/7);
figure(62)
plot(depth,error_mse_data,'*',x_depth,error_mse_ana);
ylim([0.05,0.07])
xlabel('$H_{\it\Pi}$', 'Interpreter', 'latex');
ylabel('$\tilde{\varepsilon}_{E_X,2}$', 'Interpreter', 'latex');
legend('Numerical results', ['$\tilde{\varepsilon}_{E_X,2} = ',num2str(b),'H_{\it\Pi}^{-1/7}$'], 'Interpreter', 'latex');
set(gca,'fontsize',15);
% print('Fig 1(b) Relative_error_under_L2_norm', '-depsc','-r300');
% exportgraphics(gcf,'Fig 1(b) Relative_error_under_L2_norm.pdf','Resolution',300);
% savefig('Fig 1(b) Relative_error_under_L2_norm.fig');

a = 0;
b = 0.135;
x_depth = linspace(depth(1),depth(end),50);
error_loss_ana = a+b*x_depth.^(-1/7);
figure(62)
plot(depth,error_loss_data,'*',x_depth,error_loss_ana);
% ylim([0.05,0.07])
xlabel('$H_{\it\Pi}$', 'Interpreter', 'latex');
ylabel(['$\tilde{\varepsilon}_{E_X,',num2str(loss_L),'}$'], 'Interpreter', 'latex');
legend('Numerical results', ['$\tilde{\varepsilon}_{E_X,',num2str(loss_L),'} = ',num2str(b),'H_{\it\Pi}^{-1/7}$'], 'Interpreter', 'latex');
set(gca,'fontsize',15);