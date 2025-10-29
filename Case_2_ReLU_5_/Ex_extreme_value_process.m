function Ex = Ex_extreme_value_process(X_in);
[row,col] = size(X_in);
Ex = zeros([row,col]);
for i = 1:row
    Ex(i,:) = max(abs(X_in(1:i,:)));
end
end