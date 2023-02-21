%% Set up reconstruction parameters

function [reconIms, reconTime] = reconFISTA_exp(datpath, resultpath, loadname, testInds,...
    fistaOpts, Jname, mname)

J_struct = load(sprintf("%s/%s", datpath, loadname));

avgMaxK = fistaOpts.avgMaxK;
scaleMag_J = fistaOpts.scaleMag_J;
scaleMag_m = fistaOpts.scaleMag_m;

domain = 'mvp';
waveletLvl = 3;

%%

J_mc = J_struct.(Jname);
J_mc = J_mc * (scaleMag_J ./ mean(maxk(J_mc(:), avgMaxK)));
loadStruct(J_struct.Jheaders);

VOX_W = double(VOX_W);
VOX_L = double(VOX_L);
nVox = VOX_L*VOX_W;

J_mc = reshape(J_mc, [], nVox);

sizeX = size(J_mc,2);
[f_fista, fT_fista, fistaStep] = mat2Handle(J_mc, domain,...
    'VOX_L', VOX_L, 'VOX_W', VOX_W, 'lvl', waveletLvl);

%% Process measurements

nIms = length(testInds);

m_mc = J_struct.(mname);
m = squeeze(m_mc(:,:,testInds));
m = reshape(m, [], nIms);

m = m .* (scaleMag_m ./ mean(maxk(m, avgMaxK, 1), 1));

tic; [fistaRecon, ~] = fista(m,f_fista,fT_fista,fistaStep,sizeX,fistaOpts); reconTime = toc;

reconIms = rot90(reshape(fistaRecon, [VOX_L, VOX_W, nIms]), 1);

%% Save results
truthIms = J_struct.truthIms;
savevars = {"reconIms", "truthIms", "reconTime",...
            "fistaOpts", "testInds"};

savedir = sprintf("%s/exp", resultpath);


savename = sprintf("recon_result_fista_ToF_widefield_%s", loadname);
fullsavepath = sprintf("%s/%s", savedir, savename);

if fistaOpts.shouldSave
    save(fullsavepath, savevars{:});
end

end