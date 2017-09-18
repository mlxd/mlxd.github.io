function [df] = rdf(xDim,yDim,radius,x,y,rho,mode)

if mode==0
    df = rdf_t([x'; y'],rho);
elseif mode==1
    df = rdf_c([x'; y'], radius,6.821230e-07,2.044094e-05,rho);
elseif mode==2
    df = rdf_dcc([x'; y'],radius,rho);
elseif mode==3
    df = rdf_ev([x'; y'],radius,rho);
else
    df = adf_ev([x'; y'],radius,rho)
end

end



%% Density



function [rdf_t] = rdf_t(idx, radius,rho)
    radii=1:radius

    rdf_t = zeros(length(idx),1);
    for kk=1:length(idx)

        for xx=-511:512
            for yy=-511:512
                rad(xx+512,yy+512) = sqrt((xx-(idx(1,kk)-512))^2 + (yy-(idx(2,kk)-512))^2); 
            end
        end
        rdf = zeros(radii,1);
        for ii=radii
            [r0,c0]=find( rad >= ii & rad < ii+1);
            %%rdf(ii)=0;
            for jj=length(c0)
                rdf(ii)=rdf(ii) + abs(wfc0(r0(jj),c0(jj)))*dx*dx;
            end
            rdf(ii) = rdf(ii)/(pi*((dx*ii)^2 - (dx*(ii+1))^2));
        end

        rdf_t(kk,:) = rdf;
        kk
    end
end



function [dc] = rdf_c(idx, radius,dx,d0,rho)
    %% Discrete Vortex cores
    dc=[];
    for kk=1:length(idx)
        dr=1;
        radii=(1:radius);
        count=zeros(max(radii),1);
        ref_idx = kk;
        ref(1) = idx(1,ref_idx);
        ref(2) = idx(2,ref_idx);
        for jj = radii
            count(jj) = 0;
            for ii=1:length(idx)
                indX = idx(1,ii);
                indY = idx(2,ii);
                r1 = sqrt( (0.5*d0/dx)^2 + (0.5*d0/dx)^2 );
                r2 = sqrt( ( jj )^2 + (jj)^2 );
                L = sqrt( (indX - ref(1))^2 + (indY - ref(2))^2);
                if (r1 + r2 > L || r1 + r2 + dr > L) && L>0
                   if r2 < L + r1 || r2 + dr < L + r1
                       count(jj) = count(jj) +1;
                   end
                end
            end
            %count (jj) = count(jj)/(pi*(r2-jj)^2);
            jj;
        end
        %plot(count);
        dc(kk,:)=count;
        kk
    end
end


function [dcc] = rdf_dcc(idx,radius,rho)

    %% Discrete cores, centres only
    dcc=[];
    rho = 455;
    for kk=1:length(idx)
        dr=1;
        radii=(1:dr:radius);
        count=zeros(max(radii),1);
        ref_idx = kk;
        ref(1) = idx(1,ref_idx);
        ref(2) = idx(2,ref_idx);
        for jj = radii
            count(round(jj/dr)) = 0;
            for ii=1:length(idx)
                indX = idx(1,ii);
                indY = idx(2,ii);
                r1 = sqrt( (1)^2 + (1)^2 );
                r2 = sqrt( (jj )^2 + ( jj)^2 );
                L = sqrt( (indX - ref(1))^2 + (indY - ref(2))^2);
                if (r1 + r2 > L || r1 + r2 + dr > L) && L>0
                   if r2 < L + r1 || r2 + dr < L + r1
                       count(round(jj/dr)) = count(round(jj/dr)) +1;
                   end
                end
            end
            count(round(jj/dr)) = count(round(jj/dr))/(2*rho*pi*r2*dr);
            jj;
        end
        %plot(count);
        dcc(kk,:)=count;
        kk
    end
    
end


function [dcc_ev] = rdf_ev(idx_ev,radius,rho)
    %% Discrete cores, centres only ev
    dcc_ev=zeros(length(idx_ev),512);
    rho = 1;
    for kk=1:length(idx_ev)
        dr=1;
        radii=(1:dr:radius);
        count=zeros(max(radii),1);
        ref_idx = kk;
        ref(1) = idx_ev(1,ref_idx);
        ref(2) = idx_ev(2,ref_idx);
        for jj = radii
            count(round(jj/dr)) = 0;
            for ii=1:length(idx_ev)
                indX = idx_ev(1,ii);
                indY = idx_ev(2,ii);
                r1 = sqrt( (1)^2 + (1)^2 );
                r2 = sqrt( (jj )^2 + ( jj)^2 );
                L = sqrt( (indX - ref(1))^2 + (indY - ref(2))^2);
                if (r1 + r2 > L || r1 + r2 + dr > L) && L>0
                   if r2 < L + r1 || r2 + dr < L + r1
                       count(round(jj/dr)) = count(round(jj/dr)) +1;
                   end
                end
            end
            count(round(jj/dr)) = count(round(jj/dr))/(2*rho*pi*r2*dr);
        end
        %plot(count);
        dcc_ev(kk,:)=count;
        kk;
    end
end

function [acc_ev] = adf_ev(idx_ev,rad,rho)

    %% Discrete cores, angular distribution function
    acc=[];
    rho = 1;
    rad=512;
    vorts=idx_ev;
    radii=1:rad;
    for kk=1:length(vorts)
        dtheta=0.001;
        theta=(0:dtheta:(2*pi));
        count=zeros(length(radii),1);
        ref_idx = kk;
        ref(1) = vorts(1,ref_idx);
        ref(2) = vorts(2,ref_idx);
        iii=1;
        for jj = theta
            count(iii) = 0;
            for ii=1:length(vorts)
                indX = vorts(1,ii)-ref(1);
                indY = vorts(2,ii)-ref(2);
                RPn = sqrt(indX^2 + indY^2);

                phi = atan2(indY,indX) + pi;

                if(phi >= jj && phi < jj + dtheta)
                    count(iii) = count(iii) +1;
                end

            end

            count(iii) = count(iii)/(0.5*norm(RPn).^2*dtheta);
            iii=iii+1;
        end
        kk
        acc(kk,:)=count;
    end
    figure;
    plot(0:dtheta:(2*pi),sum(acc));
    set(gca,'XTickLabels',{'0','$\pi/3$','$2\pi/3$','$\pi$','$4\pi/3$','$5\pi/3$','$2\pi$'});
    set(gca,'TickLabelInterpreter', 'latex');
    set(gca,'FontName','Latin Modern Roman','FontSize',22);
    axis square
    axis([0, 2*pi, 0, max(sum(acc))])

end
