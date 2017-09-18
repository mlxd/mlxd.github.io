function [g] = corrfunc(x,y,r,dr)
RR = sqrt(bsxfun(@minus,x,x').^2 + bsxfun(@minus,y,y').^2 );
g = zeros(300,1);
for r=1:1:300
for ii=1:length(x)
    count = 0;
    for jj=ii:length(x)
        if (RR(ii,jj) < r+dr ) & (RR(ii,jj) >= r )
           count = count + 1;
        end
    end
    g(r) = g(r) + count;
end
end