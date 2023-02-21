clear; close all; clc;

addpath("../setpaths");
[libpath, datpath, resultpath, basepath] = setpaths;

addpath("benchmark_linear_solvers");
addpath(sprintf("%s/export_fig", libpath));
addpath(sprintf("%s/subaxis", libpath));
addpath(genpath(sprintf("%s/lib", libpath)));
addpath(genpath(sprintf("%s/lib", basepath)));
addpath(genpath('./'));

datname = "allTrainingDat_30-Sep-2021";
savename_EML = sprintf("%s_EML", datname);
modelname_base = sprintf("model_%s_NL=1_nEpoch=2000_lossFunc=MAE_untied=T_vgg=F_unet_nfilts=0_act=shrink", savename_EML);
modelname_vgg = sprintf("model_%s_NL=1_nEpoch=400_lossFunc=MAE_untied=T_vgg=T_unet_nfilts=16_act=shrink", savename_EML);
modelname_fc = sprintf("model_FC");
modelname_automap = sprintf("model_automap");

fullpath_preprocess = sprintf("%s/%s.mat", datpath, savename_EML);
fullpath_model = sprintf("%s/exp/%s.mat", resultpath, modelname_base);

showInds = [6, 8, 16];

%% 1. Preprocess experimentally collected data

if ~exist(fullpath_preprocess, 'file')
    s = 0.0001;
    preprocessDatExp(datpath, datname, s, savename_EML);
end

%% 2. Perform training (for debugging information, open in jupyter)

jupyter_cmd = "jupyter nbconvert --execute --to notebook --inplace unrolled-DOT_exp_train.ipynb";

if ~exist(fullpath_model, 'file')
    system(jupyter_cmd);
end

%% 3. Run all solvers in benchmark

fistaOpts.lam1 = 1e0;
fistaOpts.lam2 = 0;
fistaOpts.maxItr = 30;
fistaOpts.tol = 0;
fistaOpts.nonneg = true; 
fistaOpts.showFigs = false;
fistaOpts.avgMaxK = 5; % Normalize Jacobian and meas to average of the top avgMaxK values 
fistaOpts.scaleMag_J = 1;
fistaOpts.scaleMag_m = 1;
fistaOpts.shouldSave = true;

admmOpts.maxIters = 70;
admmOpts.tau_inc = 1.2;
admmOpts.tau_dec = 1.2;
admmOpts.eps = 0.8;
admmOpts.lamL2 = 0;
admmOpts.mu_nneg = 0.5; % step size for non-neg reg
admmOpts.eta_nneg_init = 1e-2; % initialize Lagrange multiplier for non-neg reg
admmOpts.mu_TV = 1e-2; % step size for sparsity reg
admmOpts.gam_TV = 5e0; % Lagrange multiplier for sparsity constraint
admmOpts.eta_TV_init = 1e-2;
admmOpts.mu_L1 = 5e-2;
admmOpts.gam_L1 = 0;
admmOpts.eta_L1_init = 1e-2;
admmOpts.avgMaxK = 5; % Normalize Jacobian and meas to average of the top avgMaxK values 
admmOpts.scaleMag_J = 2.1e-4;
admmOpts.scaleMag_m = 6e5;

procDat = load(sprintf("%s/%s", datpath, savename_EML));
truthIms = procDat.truthIms;

load(fullpath_model);
testInds = testInds + 1; % converting python to matlab indices
truthIms_test = truthIms(:,:,testInds);

[reconIms_admm, reconTime_admm] = reconADMM_exp(datpath, resultpath, savename_EML, testInds, admmOpts,...
    "J_L", "diff_L");
[reconIms_fista, reconTime_fista] = reconFISTA_exp(datpath, resultpath, savename_EML, testInds, fistaOpts,...
    "J_L", "diff_L");

run_runtime_cmd_fc = sprintf("python3 support_scripts/unrolled_DOT_exp_test.py %s", modelname_fc);
[~, pyout_fc] = system(run_runtime_cmd_fc);
if startsWith(pyout_fc, "'python3' is not recognized")
    run_runtime_cmd_fc = sprintf("python support_scripts/unrolled_DOT_exp_test.py %s", modelname_fc);
    [~, pyout_fc] = system(run_runtime_cmd_fc);
end
recon_fc = load(sprintf("%s/exp/%s", resultpath, strip(pyout_fc)));

run_runtime_cmd_automap = sprintf("python3 support_scripts/unrolled_DOT_exp_test.py %s", modelname_automap);
[~, pyout_automap] = system(run_runtime_cmd_automap);
if startsWith(pyout_automap, "'python3' is not recognized")
    run_runtime_cmd_automap = sprintf("python support_scripts/unrolled_DOT_exp_test.py %s", modelname_automap);
    [~, pyout_automap] = system(run_runtime_cmd_automap);
end
recon_automap = load(sprintf("%s/exp/%s", resultpath, strip(pyout_automap)));

run_runtime_cmd_base = sprintf("python3 support_scripts/unrolled_DOT_exp_test.py %s", modelname_base);
[~, pyout_base] = system(run_runtime_cmd_base);
if startsWith(pyout_base, "'python3' is not recognized")
    run_runtime_cmd_base = sprintf("python support_scripts/unrolled_DOT_exp_test.py %s", modelname_base);
    [~, pyout_base] = system(run_runtime_cmd_base);
end
recon_unrolled_base = load(sprintf("%s/exp/%s", resultpath, strip(pyout_base)));

run_runtime_cmd_vgg = sprintf("python3 support_scripts/unrolled_DOT_exp_test.py %s", modelname_vgg);
[~, pyout_vgg] = system(run_runtime_cmd_vgg);
if startsWith(pyout_vgg, "'python3' is not recognized")
    run_runtime_cmd_vgg = sprintf("python support_scripts/unrolled_DOT_exp_test.py %s", modelname_vgg);
    [~, pyout_vgg] = system(run_runtime_cmd_vgg);
end
recon_unrolled_vgg = load(sprintf("%s/exp/%s", resultpath, strip(pyout_vgg)));

%% 4. Display reconstructions

% Normalize all images to 1
truthIms_test = max(truthIms_test ./ max(truthIms_test, [], [1,2]), 0);
reconIms_admm = max(reconIms_admm ./ max(reconIms_admm, [], [1,2]), 0);
reconIms_fista = max(reconIms_fista ./ max(reconIms_fista, [], [1,2]), 0);
reconIms_fc = max(recon_fc.reconIms_unrolled ./ max(recon_fc.reconIms_unrolled, [], [1,2]), 0);
reconIms_automap = max(recon_automap.reconIms_unrolled ./ max(recon_automap.reconIms_unrolled, [], [1,2]), 0);
reconIms_unrolled_base = max(recon_unrolled_base.reconIms_unrolled ./ max(recon_unrolled_base.reconIms_unrolled, [], [1,2]), 0);
reconIms_unrolled_vgg = max(recon_unrolled_vgg.reconIms_unrolled ./ max(recon_unrolled_vgg.reconIms_unrolled, [], [1,2]), 0);

spacing_config = {'SpacingHoriz', 0.005, 'SpacingVert', 0.005,...
        'PaddingTop', 0.00, 'PaddingLeft', 0.00, 'margin', 0.00};
f_tof = figure('Position', [40 440 1020 440]);

ncols = 7;
nrows = length(showInds);

for i = 1:nrows

    truth_i_tof = truthIms_test(:,:,showInds(i));
    admm_i_tof = reconIms_admm(:,:,showInds(i));
    fista_i_tof = reconIms_fista(:,:,showInds(i));
    fc_i = reconIms_fc(:,:,showInds(i));
    automap_i = reconIms_automap(:,:,showInds(i));
    unrolled_i_base = reconIms_unrolled_base(:,:,showInds(i));
    unrolled_i_vgg = reconIms_unrolled_vgg(:,:,showInds(i));
    
    subaxis(nrows,ncols,(i-1)*ncols+1, spacing_config{:});
    imagesc(admm_i_tof);
    set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);
    
    subaxis(nrows,ncols,(i-1)*ncols+2, spacing_config{:});
    imagesc(fista_i_tof);
    set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);

    subaxis(nrows,ncols,(i-1)*ncols+3, spacing_config{:});
    imagesc(fc_i);
    set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);
    
    subaxis(nrows,ncols,(i-1)*ncols+4, spacing_config{:});
    imagesc(automap_i);
    set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);
    
    subaxis(nrows,ncols,(i-1)*ncols+5, spacing_config{:});
    imagesc(unrolled_i_base);
    set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);

    subaxis(nrows,ncols,(i-1)*ncols+6, spacing_config{:});
    imagesc(unrolled_i_vgg);
    set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);
    
    subaxis(nrows,ncols,(i-1)*ncols+7, spacing_config{:});
    imagesc(truth_i_tof);
    set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);
end

savename = "recon_out";
fullsavepath = fullfile(resultpath, 'exp', savename);
export_fig(sprintf("%s_tofdot", fullsavepath), f_tof, '-transparent', '-m3', '-png');

%% 5 Plot results

plot_font = 'Times New Roman';
plot_fontsize = 18;

nTests = size(truthIms_test, 3);
avg_mse_admm = 0;
avg_mse_fista = 0;
avg_mse_fc = 0;
avg_mse_automap = 0;
avg_mse_unrolled_base = 0;
avg_mse_unrolled_vgg = 0;
for i = 1:nTests
    avg_mse_admm = avg_mse_admm + mse(reconIms_admm(:,:,i), truthIms_test(:,:,i));
    avg_mse_fista = avg_mse_fista + mse(reconIms_fista(:,:,i), truthIms_test(:,:,i));
    avg_mse_fc = avg_mse_fc + mse(reconIms_fc(:,:,i), truthIms_test(:,:,i));
    avg_mse_automap = avg_mse_automap + mse(reconIms_automap(:,:,i), truthIms_test(:,:,i));
    avg_mse_unrolled_base = avg_mse_unrolled_base + mse(reconIms_unrolled_base(:,:,i), truthIms_test(:,:,i));
    avg_mse_unrolled_vgg = avg_mse_unrolled_vgg + mse(reconIms_unrolled_vgg(:,:,i), truthIms_test(:,:,i));
end
avg_mse_admm = 2*avg_mse_admm / nTests; 
avg_mse_fista = 2*avg_mse_fista / nTests; 
avg_mse_fc = 2*avg_mse_fc / nTests;
avg_mse_automap = 2*avg_mse_automap / nTests;
avg_mse_unrolled_base = 2*avg_mse_unrolled_base / nTests; 
avg_mse_unrolled_vgg = 2*avg_mse_unrolled_vgg / nTests;

mse_arr = [avg_mse_admm, avg_mse_fista, avg_mse_fc, avg_mse_automap, avg_mse_unrolled_base, avg_mse_unrolled_vgg];
runtime_arr_ms = 1000*[reconTime_admm, reconTime_fista, recon_fc.reconTime_unrolled, recon_automap.reconTime_unrolled, recon_unrolled_base.reconTime_unrolled, recon_unrolled_vgg.reconTime_unrolled];

label_arr = ["ADMM", "FISTA", "FC", "Automap", "Unrolled-DOT (base)", "Unrolled-DOT (UNet+VGG)"];

% disp(mse_arr);
% disp(runtime_arr_ms);

% dx = [+0.5, +0.7, +0.5, -1.2];
% dy = [-30000,-11000,-1000,+30];

f = figure('Position', [970, 440, 770, 440]); 
scatter(mse_arr(1:(end-2)), runtime_arr_ms(1:(end-2)), 100, 'filled');
hold on;
scatter(mse_arr((end-1):end), runtime_arr_ms((end-1):end), 100, 'filled');
% text(mse_arr+dx, runtime_arr_ms+dy, label_arr, 'HorizontalAlignment', 'center','FontSize', plot_fontsize, 'FontName', plot_font);
% title("Reconstruction time vs MSE");
ylabel("Runtime (ms)");
xlabel("MSE");

ax_ticks = [1e-2, 2.5e-2, 6e-2, 15e-2, 30e-2];
ax_ticks_cells = cellstr(string(ax_ticks));

set(gca, 'XTick', ax_ticks);
set(gca, 'XTickLabels', ax_ticks_cells);
set(gca, 'YScale', 'log');
set(gca, 'XScale', 'log');
set(gca,'FontSize', plot_fontsize, 'FontName', plot_font);
yplot_min = min(runtime_arr_ms) ./ 5;
yplot_max = max(runtime_arr_ms) .* 5;
ylim([yplot_min, yplot_max]);
xlim([9e-3, 325e-3]);

export_fig(f, sprintf("%s/exp/recon_result", resultpath), '-png', '-m3', '-transparent');

%%

savedir = fullfile(resultpath, 'exp');
final_report_str = "";
for i = 1:length(label_arr)
    str_line = sprintf("%s reconstruction. MSE: %.3d; Runtime: %.3d\n", label_arr(i), mse_arr(i), runtime_arr_ms(i));
    fprintf(str_line);
    final_report_str = sprintf("%s%s", final_report_str, str_line);
end

report_filename = "avg_mse_runtime.txt";
fid = fopen(fullfile(resultpath, 'exp', report_filename), 'w');
fprintf(fid, final_report_str);


