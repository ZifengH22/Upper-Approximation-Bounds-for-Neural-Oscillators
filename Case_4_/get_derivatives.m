function dy = get_derivatives(y, M_diag, C, k, alpha, xy, N, ag_t)
    x = y(1:N);
    v = y(N+1:2*N);       % 加上了 *
    xp = y(2*N+1:3*N);    % 加上了 *
    
    % 1. 计算层间位移和层间速度 (Interstory drifts and velocities)
    dx = [x(1); diff(x)];
    dv = [v(1); diff(v)];
    
    % 2. 判断屈服状态并更新塑性流动 (Plastic flow rule for kinematic hardening)
    dxp = zeros(N, 1);
    % 屈服条件: 当层间弹性变形超过屈服位移，且运动方向与塑性变形方向一致
    yield_condition = (abs(dx - xp) >= xy) & (sign(dx - xp) == sign(dv));
    dxp(yield_condition) = dv(yield_condition); % 屈服时，塑性变形率 = 层间速度
    
    % 3. 计算各个弹簧的非线性恢复力
    fs = alpha * k .* dx + (1 - alpha) * k .* (dx - xp);
    
    % 4. 计算作用在各个质量块上的合力 (F_mass_i = fs_i - fs_i+1)
    F_mass = fs;
    F_mass(1:N-1) = F_mass(1:N-1) - fs(2:N);
    
    % 5. 运动方程求解加速度 dv/dt = -M^{-1}(C*v + F_mass) - ag
    % 因为是集中质量矩阵，M为对角阵，可直接用点除加速
    dvdt = - (C * v + F_mass) ./ M_diag - ag_t;
    
    % 6. 拼装导数向量
    dy = [v; dvdt; dxp];
end