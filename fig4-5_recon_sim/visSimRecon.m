%% Load unrolled DOT results
clear; close all; clc;

addpath("../setpaths");
[libpath, datpath, resultpath, basepath] = setpaths;

addpath(sprintf("%s/export_fig", libpath));
addpath(sprintf("%s/subaxis", libpath));

loaddir = sprintf("%s/sim", resultpath);

sim_params = [{'f', 'f', false}; {'f', 'f', true}; {'f', 'm', false}; {'m', 'm', false}];

final_report_str = "";

for i = 1:length(sim_params)

sim_params_i = sim_params(i,:);
train_str = sim_params_i{1};
test_str = sim_params_i{2};
vgg_unet = sim_params_i{3};

if strcmp(train_str, 'f')
    name_str = "Trained on Fashion";
else
    name_str = "Trained on MNIST";
end
if strcmp(test_str, 'f')
    name_str = sprintf("%s; Tested on Fashion", name_str);
else
    name_str = sprintf("%s; Tested on MNIST", name_str);
end

if vgg_unet
    vgg_str = 'T';
    nfilts_unet = 32;
    n_epochs = 200;
    name_str = sprintf("%s; with vgg u-net", name_str);
else
    vgg_str = 'F';
    nfilts_unet = 0;
    n_epochs = 2000;
    name_str = sprintf("%s; WITHOUT vgg u-net", name_str);
end



if strcmp(train_str, 'f') && strcmp(test_str, 'f')
    loss_str = 'MSE';
else
    loss_str = 'MAE';
end

unrolled_name = sprintf("model_5_1_22_unrolled_jac_train=%s_test=%s_NL=3_nEpoch=%d_lossFunc=%s_untied=T_vgg=%s_unet_nfilts=%d", train_str, test_str, n_epochs, loss_str, vgg_str, nfilts_unet);
full_loadname = sprintf("%s/%s.mat", loaddir, unrolled_name);

load(full_loadname);

if strcmp(train_str, 'f') && strcmp(test_str, 'f')
    truth_train_norm = truth_train_np;
    recon_train_norm = recon_train_np;
    truth_test_norm = truth_test_np;
    recon_test_norm = recon_test_np;
else
    truth_train_norm = max(truth_train_np ./ max(truth_train_np, [], [1, 2]), 0);
    recon_train_norm = max(recon_train_np ./ max(recon_train_np, [], [1, 2]), 0);
    truth_test_norm = max(truth_test_np ./ max(truth_test_np, [], [1, 2]), 0);
    recon_test_norm = max(recon_test_np ./ max(recon_test_np, [], [1, 2]), 0);
end

plotInds = [2, 10, 14, 41];


nTests = size(recon_test_norm,3);
avg_mse_test = 0;
avg_corr_coeff = 0;
for k = 1:nTests
    avg_mse_test = avg_mse_test + 2*mse(recon_test_norm(:,:,k), truth_test_norm(:,:,k));
    corr_coeff_k = corrcoef(recon_test_norm(:,:,k), truth_test_norm(:,:,k));
    avg_corr_coeff = avg_corr_coeff + corr_coeff_k(1,2);
end
avg_mse_test = avg_mse_test / nTests;
avg_corr_coeff = avg_corr_coeff / nTests;


mse_report_str = sprintf("%s; Average Test Set MSE: %d, Correlation Coefficient: %.4f\n", name_str, avg_mse_test, avg_corr_coeff);
final_report_str = sprintf("%s%s", final_report_str, mse_report_str);
fprintf(mse_report_str);

spacing_config1 = {'SpacingHoriz', 0.005, 'SpacingVert', 0.005,...
        'PaddingTop', 0.00, 'PaddingLeft', 0.00, 'margin', 0.00};
spacing_config2 = {'SpacingHoriz', 0.005, 'SpacingVert', 0.005,...
        'PaddingTop', 0.02, 'PaddingLeft', 0.00, 'margin', 0.00};

nrows = 2;
ncols = length(plotInds);

ncols_truth_only = 2*ncols;
nrows_truth_only = 1*nrows;
f_train_truth_only = figure('Position', [845 270 610 150]);
for k = 1:ncols_truth_only
    for j = 1:nrows_truth_only
        truth_i = sub2ind([nrows_truth_only, ncols_truth_only], j, k);
        subaxis(nrows_truth_only,ncols_truth_only,truth_i, spacing_config2{:});
        imagesc(truth_train_np(:,:,truth_i));
        set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);
    end
end

f_train = figure('Position', [845 270 610 305]);
for k = 1:ncols
    subaxis(nrows,ncols,k, spacing_config2{:});
    imagesc(truth_train_norm(:,:,plotInds(k)));
    set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);
    
    subaxis(nrows,ncols,ncols*1+k, spacing_config1{:});
    imagesc(recon_train_norm(:,:,plotInds(k)));
    set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);
end

f_test = figure('Position', [845 270 610 305]);
for k = 1:ncols
    subaxis(nrows,ncols,k, spacing_config2{:});
    imagesc(truth_test_norm(:,:,plotInds(k)));
    set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);
    
    subaxis(nrows,ncols,ncols*1+k, spacing_config1{:});
    imagesc(recon_test_norm(:,:,plotInds(k)));
    set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]);
end

savedir = sprintf("%s/%s", loaddir, unrolled_name);
if ~exist(savedir, 'dir')
    mkdir(savedir);
end

savename_train = "recon_train";
fullsavepath_train = fullfile(savedir, savename_train);
export_fig(fullsavepath_train, f_train, '-transparent', '-m3', '-png');

savename_test = "sim_recon_test";
fullsavepath_test = fullfile(savedir, savename_test);
export_fig(fullsavepath_test, f_test, '-transparent', '-m3', '-png');

savename_train_truth_only = "sim_recon_train_truth_only";
fullsavepath_train_truth_only = fullfile(savedir, savename_train_truth_only);
export_fig(fullsavepath_train_truth_only, f_train_truth_only, '-transparent', '-m3', '-png');

end

report_filename = "avg_mse.txt";
fid = fopen(fullfile(loaddir, report_filename), 'w');
fprintf(fid, final_report_str);

