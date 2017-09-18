
function [] = track_tails(vorts)
vorts2=vorts(1:end-18,2:end,:);
ravg = zeros(300,1);
    figure;
    aa=zeros(300);
for rbin=1:1:500

    r_idx=find ( sqrt( (vorts2(:,1,1)-512).^2 + (vorts2(:,1,2)-512).^2 ) >= rbin-1 & sqrt( (vorts2(:,1,1)-512).^2 + (vorts2(:,1,2)-512).^2 ) < rbin );

    if (~isempty(r_idx))
        for ii=1:length(r_idx)
           diffs = (abs((diff(sqrt( (vorts2(r_idx(ii),:,1)).^2 + (vorts2(r_idx(ii),:,2)).^2 )))));
           %diffs(isnan(diffs)) = 0;
           ravg(rbin) = ravg(rbin) + sum(diffs);
           %Total distance traveled
           plot(rbin,sum(diffs),'r*');hold on
           
           
           diffs2 = (((diff(sqrt( (vorts2(r_idx(ii),:,1)).^2 + (vorts2(r_idx(ii),:,2)).^2 )))));
           %diffs(isnan(diffs)) = 0;
           ravg(rbin) = ravg(rbin) + sum(diffs);
           %Actual distance traveled
           %plot(rbin,sum(diffs2),'k*');hold on
           
           
           %Linear distance traveled
           plot(rbin, sqrt( (vorts2(r_idx(ii),1,1) - vorts2(r_idx(ii),end,1)).^2 + (vorts2(r_idx(ii),1,2) - vorts2(r_idx(ii),end,2)).^2 ),'bo');hold on
        %Arc distance traveled -> not working correctly, check angles
           a=getAngle([ squeeze(vorts2(r_idx(ii),1,:))'; squeeze(vorts2(r_idx(ii),end,:))']);
           aa(rbin*3 + ii) = a;
           plot(rbin, (a).*sqrt( (vorts2(r_idx(ii),1,1) - 512).^2 + (vorts2(r_idx(ii),1,2) - 512).^2),'gx');hold on
        end
        ravg(rbin) = ravg(rbin)./length(r_idx);
    end 
end
legend('Total dist','Actual dist.','Lin. dist','Arc dist');
title(pwd)

%% Averaged values
%figure;plot(ravg(find(ravg>0)),'g-')
%figure;plot(ravg,'c-')
%figure;plot(aa,'r*')
