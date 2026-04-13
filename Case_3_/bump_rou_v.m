function rou_v = bump_rou_v(x,v,dt)
v = v(:);

[X, V] = meshgrid(x, v);

rou_v = zeros(size(X));

idx = (X > -V) & (X < 0);

temp = exp( (-V.^2)./(V.^2 - (2*X + V).^2) );

rou_v(idx) = temp(idx);

rou_v = rou_v./(sum(rou_v,2)*dt);

end