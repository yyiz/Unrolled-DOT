%% Set up reconstruction parameters

function [reconIms, reconTime] = reconADMM_exp(datpath, resultpath, loadname, testInds,...
    admmOpts, Jname, mname)

J_struct = load(sprintf("%s/%s", datpath, loadname));

avgMaxK = admmOpts.avgMaxK; 
scaleMag_J = admmOpts.scaleMag_J;
scaleMag_m = admmOpts.scaleMag_m;

%% Pre-process data

J_mc = J_struct.(Jname);
J_mc = J_mc * (scaleMag_J ./ mean(maxk(J_mc(:), avgMaxK)));
loadStruct(J_struct.Jheaders);

VOX_W = double(VOX_W);
VOX_L = double(VOX_L);
nVox = VOX_L*VOX_W;

J_mc = reshape(J_mc, [], nVox);

%% Process measurements

nIms = length(testInds);

m_L = J_struct.(mname);
m = squeeze(m_L(:,:,testInds));
m = reshape(m, [], nIms);

m = m .* (scaleMag_m ./ mean(maxk(m, avgMaxK, 1), 1));

%% Call ADMM

[Dx, Dy] = genDx(VOX_W, VOX_L, VOX_W, VOX_L); % Generate gradient matrices

tic;
[reconAdmm, ~] = admm_MV(J_mc, m, Dx, Dy, admmOpts);
reconTime = toc;

%% Post-processing and plotting

% Plot FISTA image reconstruction
reconIms = rot90(reshape(reconAdmm, [VOX_L, VOX_W, nIms]), 1);

%% Save results
truthIms_test = J_struct.truthIms(:,:,testInds);
savevars = {"reconIms", "truthIms_test", "reconTime",...
            "admmOpts", "testInds"};

savedir = sprintf("%s/exp", resultpath);
savename = sprintf("recon_result_admm_widefield_%s", loadname);
fullsavepath = sprintf("%s/%s", savedir, savename);

save(fullsavepath, savevars{:});

end
