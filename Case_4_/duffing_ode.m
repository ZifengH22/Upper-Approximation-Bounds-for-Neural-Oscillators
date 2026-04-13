function dy = duffing_ode(y, M_diag, C, k_lin, beta, N, ag_t)
    x = y(1:N);
    v = y(N+1:2*N);
    
    % Calculate relative interstory drifts
    dx = [x(1); diff(x)];
    
    % Duffing Restoring Force: F_s = k*dx + beta*dx^3
    % Assuming every story has the same Duffing characteristic
    fs = k_lin * dx + beta * dx.^3;
    
    % Convert story forces to nodal forces (F_nodal_i = fs_i - fs_i+1)
    F_spring = fs;
    F_spring(1:end-1) = F_spring(1:end-1) - fs(2:end);
    
    % Equation of motion: dv/dt = -M^-1 * (C*v + F_spring) - ag
    % Fast point-wise division for diagonal mass matrix
    dvdt = -(C*v + F_spring) ./ M_diag - ag_t;
    
    % Assemble derivative vector
    dy = [v; dvdt];
end