function nonlinear_reference(folder)
% Execute original loop with preinitialized identical inputs. No source edits.
source = splitlines(fileread(fullfile(fileparts(mfilename('fullpath')), '..', '..', 'KlyC', 'KLJCmainsv22.m')));
assert(contains(source{212}, 'for nn=1:Nlimit') && contains(source{532}, 'effend=interp1'));
assert(contains(source{601}, 'Yflag~=length(Y)') && strcmp(strtrim(source{626}), 'end'));
% Same syntax-only adapter as test/6.spacecharge_test: keep .* on one line.
source{322} = strrep(source{322}, 'gama/gama. ...', 'gama/gama ...');
source{323} = strrep(source{323}, '*(exp(-2.40483*1', '.*(exp(-2.40483*1');
source{321} = strrep(source{321}, 'parfor', 'for'); % Serial comparison, no pool.
source{215} = ['skipped_iterations=[skipped_iterations,nn]; ' source{215}];
record_code = ["records(end+1,:)=[nn,effend,residential,residentialel];"; ...
               "Vnc_history(end+1,:)=Vnc; Vnp_history(end+1,:)=Vnp;"; ...
               "drivefactors(end+1)=max(sqrt(min(1/Npin*(floor(2*nn/Nlimit*Npin)+1),1)),abs(sign(abs(moviedisp(4))-2)));" ];
% Omit GUI/printing/power reporting (535--600); retain original decisions.
block = strjoin([source(210:216); {'SCmethods(end+1)=SCmethod;'}; ...
                 source(217:534); cellstr(record_code); source(601:626)], newline);
files = dir(fullfile(folder, '*_input.mat'));
for case_index = 1:length(files)
    load(fullfile(folder, files(case_index).name));
    records=[]; Vnc_history=[]; Vnp_history=[]; drivefactors=[];
    SCmethods=[]; skipped_iterations=[];
    evalc(block);
    result_file = fullfile(folder, strrep(files(case_index).name, '_input.mat', '_matlab.mat'));
    save(result_file, 'Iz','QQ','IzDC','QQDC','effe','vez','gamaz', ...
        'Pinflag','Pin','Vin','flagVnc','Vnp','Vnc','Mc','B', ...
        'Zn','Vn','Tn','Tnz','Vnz','Y','Yflag','Utn','ttn','Pe','IzDCr','Izr','QQDCr','QQr', ...
        'Emz','Emz0','Esmz','Esz0','nn','Vnc_history','Vnp_history','drivefactors', ...
        'SCmethods','skipped_iterations','SCmethod','iterationsuc','effendflag','records','-v7');
    fprintf('MATLAB %s: %d attempted, %d evaluated, success=%d\n', files(case_index).name, nn, size(records,1), iterationsuc);
end
end
