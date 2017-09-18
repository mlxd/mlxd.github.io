theta1=0;
A = cos(cosd(theta1)*(xm1) - sind(theta1)*(ym1)).^2;
theta2=90;
B = cos(cosd(theta2)*(xm1) - sind(theta2)*(ym1)).^2;



for ii=0:.1:90
theta1=ii;
A1 = cos(cosd(theta1)*(xm1) - sind(theta1)*(ym1)).^2;
theta2=90+ii;
B1 = cos(cosd(theta2)*(xm1) - sind(theta2)*(ym1)).^2;

 pcolor(A+B+A1+B1);axis square;axis off; shading interp;drawnow
end