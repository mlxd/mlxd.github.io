function [vorts] = loadVtx161(start, step, fin)
%loads the processed vorts from the output of vort.py, and structures them
%into Matlab 3D matrices for Vortex, Time, X/Y

    %vorts=zeros(1,1,2); 
    a=zeros(2,1);
    steps=start:step:fin;
    
    for ii=steps
        count((ii/1000)+1)=0;
        f=csvread(strcat(strcat('vort_arr_',int2str(ii)),''),1,0);
        for jj=1:(length(f))
            if ( round(f(jj,1) == 0) || round(f(jj,2) == 0)) 
                0;
            else
                count(ii/1000 +1 ) = count(ii/1000 +1) +1;
                vorts(count(ii/1000 +1),ii/1000 +1).x = f(jj,2);
                vorts(count(ii/1000 +1),ii/1000 +1).y = f(jj,4);
                vorts(count(ii/1000 +1),ii/1000 +1).charge = round(f(jj,5));
            end
        end
    end
end