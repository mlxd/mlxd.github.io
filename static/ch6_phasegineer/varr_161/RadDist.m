function [ dcc ] = RadDist( x, y, r_x, r_y, rho )
%RadDist Calculates the radial distribution function of the given
%positions
%   x, y are vectors of x and y position. r_x, r_y are the radii of the
%   specific particles. rho is the density/unit area
    dcc=zeros(length(x),512);
    rho = 1;
    idx_ev=[x, y]';
    for kk=1:length(x)
        dr=1;
        radii=(1:dr:512);
        count=zeros(max(radii),1);
        for jj = radii
            count(round(jj/dr)) = 0;
            for ii=1:length(idx_ev)
                indX = idx_ev(1,ii);
                indY = idx_ev(2,ii);
                r1 = sqrt( (r_x)^2 + (r_y)^2 );
                r2 = sqrt( (jj )^2 + ( jj)^2 );
                L = sqrt( (indX - idx_ev(1,kk))^2 + (indY - idx_ev(2,kk))^2);
                if L>0 && (r1 + r2 > L || r1 + r2 + dr > L) 
                   if r2 < L + r1 || r2 + dr < L + r1
                       count(round(jj/dr)) = count(round(jj/dr)) +1;
                   end
                end
            end
            %count(round(jj/dr)) = count(round(jj/dr))/(2*rho*pi*r2*dr);
            count(round(jj/dr)) = pi*300*300*count(round(jj/dr))/(341*2*rho*pi*r2*dr);

        end
        dcc(kk,:)=count;
    end
end

