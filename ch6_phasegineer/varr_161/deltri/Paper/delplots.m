for ii=1:length(VarName9)
clf;
x1=[vtx{1,:,VarName9(ii)}]; y1=[vtx{2,:,VarName9(ii)}]; %q1=[vtx{3,:,ii}];
defectTriangulation(x1,y1,dx);
axis([-0.2 0.2 -0.2 0.2].*1e-3);
set(gcf,'color','black')
set(gcf,'color','none')
set(gcf,'color','white')
axis off
set(gcf,'color','black')
export_fig(['./tri_',int2str(VarName9(ii)*1000),'.png'],'-r300');
%print('-dpng','-r300',['./deltri/tri_',int2str((ii)*1000),'.png']);
end