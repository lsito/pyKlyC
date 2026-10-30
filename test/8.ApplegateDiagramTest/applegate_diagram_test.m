%% 1. Applegate diagram and harmonics: independent MATLAB plot preparation
% Run applegate_diagram_test.ipynb first, then this Live Script, then compare_results.ipynb.
% Each language reads its OWN final solver checkpoint from test 7.
% Copied calculations: KLJCmainsv22.m 708-712 and 729-773.
% Adapters: capture plot handles for comparison, omit unreadable source comments,
% add gray grids, and save outputs. No original phase/plot arithmetic is changed.
test_directory=fileparts(mfilename('fullpath'));
if ~isfile(fullfile(test_directory,'applegate_diagram_test.m'))
    test_directory=fileparts(matlab.desktop.editor.getActiveFilename);
    if ~isfile(fullfile(test_directory,'applegate_diagram_test.m'))
        test_directory=fileparts(which('applegate_diagram_test.m'));
    end
end
input_file=fullfile(test_directory,'inputs.mat');
assert(isfile(input_file),'Run the Python notebook first. Missing: %s',input_file);
controls=load(input_file);
source_directory=fullfile(fileparts(test_directory),'7.nonlinear_iteration_test');
assert(isfile(fullfile(source_directory,'matlab_results.mat')),'Run the MATLAB test 7 producer first.');
load(fullfile(source_directory,'inputs.mat'));
assert(strcmp(run_id,controls.source_run_id),'Test 7 inputs changed; rerun test 8.');
load(fullfile(source_directory,'matlab_results.mat'));
assert(strcmp(run_id,controls.source_run_id),'Stale MATLAB test 7 result.');
source_run_id=run_id;
run_id=controls.run_id;
GIFpara=controls.GIFpara;
figuredisp='on';

%% 2. Reference electron: original MATLAB lines 708-712
[mtemp,ntemp]=max(abs(Iz(:,1)));
fprintf('the deepest modulation |Iz1/Iz0|=');fprintf(num2str(mtemp));fprintf(' occurs at z=');fprintf('%6.4f mm\n',zi(ntemp));
Izmax=mtemp;
[~,ncenter]=min(abs(exp(1i*Tnz(ntemp,Y))-exp((pi/2-angle(Iz(ntemp,1)))*1i)));
ncenter=Y(ncenter);

%% 3. Original first two plots: MATLAB lines 729-773
clear xlim;
% Clear only the two numbered figures this test will use.
figure(1); clf;
figure(2); clf;
trajectory_handles=gobjects(0);
harmonic_handles=gobjects(0);
cavity_handles=gobjects(0);
ez=ez.*repmat(sign(nh),length(zi),1);% DC field will not be displayed as background
if strcmp(figuredisp,'on')
    zmin=min(zi)+GIFpara(4); % lower limit for the position
    zmax=max(zi)-GIFpara(5); % upper limit for the position
    ymin=GIFpara(2); % lowest ve/c plot on the fig
    ymax=GIFpara(3); % highest ve/c plot on the fig
    velocitytype=GIFpara(6); % 0--real velocity/c; other numbers--velocity/c*gama
    Ntp=GIFpara(1); % time steps in one RF period
    
    figure(1);
    [AX]=plotyy(z(1)/betae,tn1(ncenter)-z(1),zi,abs(sum(ez,2)));
    hold on;
    
    for kp=intersect(Y,1:max(round(Nt/Ntp),1):Nt)
        trajectory_handles(end+1)=plot(Zn(:,kp)/betae,Tn(:,kp)-Zn(:,kp)-floor((Tn(:,kp)-Zn(:,kp)-tn1(ncenter)+z(1)+pi)/2/pi)*2*pi,'b.','MarkerSize',2');
        hold on;
    end
    title('applegate diagram');
    set(get(gca,'xlabel'),'string','z position /mm');
    set(AX(:),'xlim',[min(zi) zmax]);
    set(get(AX(1),'Ylabel'),'string','w*(t-z/ve)');
    set(get(AX(2),'Ylabel'),'string','ez /mm^{(-3/2)}');
    set(AX(1),'ylim',[tn1(ncenter)-z(1)-pi,tn1(ncenter)-z(1)+pi],'yTick',[tn1(ncenter)-z(1)-pi:2*pi:tn1(ncenter)-z(1)+pi]);
    set(AX(2),'ylim',[min(abs(sum(ez,2))),max(abs(sum(ez,2)))]);
    
    
    
    
    figure(2);
    for k=1:Ns
        harmonic_handles(end+1)=plot(zi,abs(Iz(:,k)),'LineWidth',2^(3-k));
        text_strings{k}=[num2str(k),'th modulation depth'];
        hold on;
    end
    for k=1:Nc
        cavity_handles(end+1)=plot([zshift(k),zshift(k)],[0,2],'r--','LineWidth',1);
        hold on;
    end
    legend(text_strings,'Location','NorthWest');
    title('modulation depth for nth harmonic current');
    xlim([min(zi) zmax]);
    xlabel('z position /mm');
    ylabel('|Izn/Iz0|');
    grid on;
    hold on;
end

%% 4. Read the actual plotted coordinates and export them
% The original plot includes inf/NaN placeholders. Represent all omitted
% history points by NaN in BOTH exported coordinates for comparison.
stride=max(round(Nt/Ntp),1);
selected=intersect(Y,1:stride:Nt);
x_phase=nan(size(Zn,1),length(selected));
phase=x_phase;
for j=1:length(selected)
    xx=get(trajectory_handles(j),'XData');
    yy=get(trajectory_handles(j),'YData');
    valid=isfinite(xx) & isfinite(yy);
    x_phase(valid,j)=xx(valid);
    phase(valid,j)=yy(valid);
end
background_handle=findobj(AX(2),'Type','line');
ez_background=get(background_handle(1),'YData');
harmonic_amplitudes=zeros(length(zi),Ns);
harmonic_widths=zeros(1,Ns);
for k=1:Ns
    harmonic_amplitudes(:,k)=get(harmonic_handles(k),'YData');
    harmonic_widths(k)=get(harmonic_handles(k),'LineWidth');
end
cavity_x=zeros(Nc,2);
cavity_y=zeros(Nc,2);
for k=1:Nc
    cavity_x(k,:)=get(cavity_handles(k),'XData');
    cavity_y(k,:)=get(cavity_handles(k),'YData');
end
xlim=get(AX(1),'XLim');
phase_ylim=get(AX(1),'YLim');
field_ylim=get(AX(2),'YLim');
harmonic_xlim=get(get(harmonic_handles(1),'Parent'),'XLim');
save(fullfile(test_directory,'matlab_results.mat'), ...
    'run_id','source_run_id','Izmax','ntemp','ncenter','selected','stride', ...
    'zi','x_phase','phase','ez_background','harmonic_amplitudes','harmonic_widths', ...
    'cavity_x','cavity_y','xlim','phase_ylim','field_ylim','harmonic_xlim','-v7');
% Formatting only: both implementations use a dashed gray grid.
for figure_number=1:2
    axes_list=findall(figure(figure_number),'Type','axes');
    set(axes_list,'XGrid','on','YGrid','on','GridLineStyle','--','GridColor',[.5 .5 .5]);
end
set(AX(2),'XGrid','off','YGrid','off');
exportgraphics(figure(1),fullfile(test_directory,'matlab_applegate.png'),'Resolution',150);
exportgraphics(figure(2),fullfile(test_directory,'matlab_current_harmonics.png'),'Resolution',150);
disp('Saved MATLAB plot data and PNGs. Next run compare_results.ipynb.');
