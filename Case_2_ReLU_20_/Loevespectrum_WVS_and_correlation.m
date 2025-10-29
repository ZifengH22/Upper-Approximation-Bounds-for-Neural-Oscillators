function [S,R,WVS] = Loevespectrum_WVS_and_correlation(t,f,theta);
f = f(:);
t = t(:)';

WVS = theta(1)*f.^2.*t.^2.*exp(-(theta(2)+theta(2)*f.^2).*t);

fmat = 0.5*(f+f');
ybmat = -f+f';
S1 = 2*theta(1)*fmat.^2; 
S2 = (fmat.^2+1).^3*theta(2)^3;
S3 = 6*sqrt(-1)*(fmat.^2+1).^2*pi.*ybmat*theta(2)^2;
S4 = -12*pi^2*ybmat.^2.*(fmat.^2+1)*theta(2);
S5 = -8*sqrt(-1)*pi^3*ybmat.^3;
Sdown = (S2+S3+S4+S5);
S = S1./Sdown;

tmat = 0.5*(t'+t);
taomat = -t'+t;
R1 = exp(-(theta(2)^2*tmat.^2+pi^2*taomat.^2)./theta(2)./tmat);
R2 = theta(1)*sqrt(pi)*(-2*pi^2*taomat.^2+theta(2)*tmat);
R3 = 2*(theta(2)^2.5)*tmat.^0.5;
R = R1.*R2./R3;

end