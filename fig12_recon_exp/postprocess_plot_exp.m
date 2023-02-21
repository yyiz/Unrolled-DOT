clear; close all; clc;

addpath("../setpaths");
[libpath, datpath, resultpath, basepath] = setpaths;

addpath("benchmark_linear_solvers");
addpath(sprintf("%s/export_fig", libpath));
addpath(sprintf("%s/subaxis", libpath));
addpath(genpath(sprintf("%s/lib", libpath)));
addpath(genpath(sprintf("%s/lib", basepath)));
addpath(genpath('./'));

showInds = [6, 8, 16];

%% 1. Load data

admm_path = sprintf("%s/exp/recon_result_admm_widefield_allTrainingDat_30-Sep-2021_EML", resultpath);
fista_path = sprintf("%s/exp/recon_result_fista_ToF_widefield_allTrainingDat_30-Sep-2021_EML", resultpath);
fc_path = sprintf("%s/exp/unrolled_dot_exp_eml_test_model_FC", resultpath);
automap_path = sprintf("%s/exp/unrolled_dot_exp_eml_test_model_automap", resultpath);
unrolled_base_path = sprintf("%s/exp/unrolled_dot_exp_eml_test_model_allTrainingDat_30-Sep-2021_EML_NL=1_nEpoch=2000_lossFunc=MAE_untied=T_vgg=F_unet_nfilts=0_act=shrink", resultpath);
unrolled_vgg_path = sprintf("%s/exp/unrolled_dot_exp_eml_test_model_allTrainingDat_30-Sep-2021_EML_NL=1_nEpoch=400_lossFunc=MAE_untied=T_vgg=T_unet_nfilts=16_act=shrink", resultpath);

admm_struct = load(admm_path);
fista_struct = load(fista_path);
recon_fc = load(fc_path);
recon_automap = load(automap_path);
recon_unrolled_base = load(unrolled_base_path);
recon_unrolled_vgg = load(unrolled_vgg_path);

truthIms_test = admm_struct.truthIms_test;

reconIms_admm = admm_struct.reconIms;
reconTime_admm = admm_struct.reconTime;

reconIms_fista = fista_struct.reconIms;
reconTime_fista = fista_struct.reconTime;

%% 2. Display reconstructions

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

%% 3. Plot results

plot_font = 'Times New Roman';
plot_fontsize = 18;

admm_sym = "+";
fista_sym = "^";
fc_sym = "o";
automap_sym = "square";
unrolled_base_sym = "pentagram";
unrolled_vgg_sym = "diamond";
sym_sz = 200;
star_sz = 800;
diamond_sz = 300;
sym_line_width = 3;

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

admm_ssim = mean(ssim(reconIms_admm, truthIms_test, 'DataFormat', 'SSB'));
fista_ssim = mean(ssim(reconIms_fista, truthIms_test, 'DataFormat', 'SSB'));
fc_ssim = mean(ssim(reconIms_fc, truthIms_test, 'DataFormat', 'SSB'));
automap_ssim = mean(ssim(reconIms_automap, truthIms_test, 'DataFormat', 'SSB'));
unrolled_base_ssim = mean(ssim(reconIms_unrolled_base, truthIms_test, 'DataFormat', 'SSB'));
unrolled_vgg_ssim = mean(ssim(reconIms_unrolled_vgg, truthIms_test, 'DataFormat', 'SSB'));

reconTime_admm_ms = 1000*reconTime_admm;
reconTime_fista_ms = 1000*reconTime_fista;
reconTime_fc_ms = 1000*recon_fc.reconTime_unrolled;
reconTime_automap_ms = 1000*recon_automap.reconTime_unrolled;
reconTime_unrolled_base_ms = 1000*recon_unrolled_base.reconTime_unrolled;
reconTime_unrolled_vgg_ms = 1000*recon_unrolled_vgg.reconTime_unrolled;

f = figure('Position', [970, 440, 770, 440]); 
yyaxis left;
hold on;
scatter(reconTime_admm_ms, admm_ssim, sym_sz, admm_sym, 'LineWidth', sym_line_width);
scatter(reconTime_fista_ms, fista_ssim, sym_sz, fista_sym, 'filled');
scatter(reconTime_fc_ms, fc_ssim, sym_sz, fc_sym, 'filled');
scatter(reconTime_automap_ms, automap_ssim, sym_sz, automap_sym, 'filled');
scatter(reconTime_unrolled_base_ms, unrolled_base_ssim, star_sz, unrolled_base_sym, 'filled', 'MarkerEdgeColor', 'y', 'LineWidth', 2);
scatter(reconTime_unrolled_vgg_ms, unrolled_vgg_ssim, diamond_sz, unrolled_vgg_sym, 'filled', 'MarkerEdgeColor', 'y', 'LineWidth', 2);
ylabel("SSIM");
ylim([0.1, 1.0]);
yyaxis right;
hold on;
scatter(reconTime_admm_ms, avg_mse_admm, sym_sz, admm_sym, 'LineWidth', sym_line_width);
scatter(reconTime_fista_ms, avg_mse_fista, sym_sz, fista_sym, 'filled');
scatter(reconTime_fc_ms, avg_mse_fc, sym_sz, fc_sym, 'filled');
scatter(reconTime_automap_ms, avg_mse_automap, sym_sz, automap_sym, 'filled');
scatter(reconTime_unrolled_base_ms, avg_mse_unrolled_base, star_sz, unrolled_base_sym, 'filled', 'MarkerEdgeColor', 'y', 'LineWidth', 2);
scatter(reconTime_unrolled_vgg_ms, avg_mse_unrolled_vgg, diamond_sz, unrolled_vgg_sym, 'filled', 'MarkerEdgeColor', 'y', 'LineWidth', 2);
set(gca, 'YScale', 'log');
set(gca, 'XScale', 'log');
xlabel("Runtime (ms)");
ylabel("MSE");
set(gca,'FontSize', plot_fontsize, 'FontName', plot_font);
ylim([5e-3, 1e0]);

export_fig(f, sprintf("%s/exp/recon_result", resultpath), '-png', '-m3', '-transparent');

%% 5. Export metrics

label_arr = ["ADMM", "FISTA", "FC", "Automap", "Unrolled-DOT (base)", "Unrolled-DOT (UNet+VGG)"];
ssim_arr = [admm_ssim, fista_ssim, fc_ssim, automap_ssim, unrolled_base_ssim, unrolled_vgg_ssim];
mse_arr = [avg_mse_admm, avg_mse_fista, avg_mse_fc, avg_mse_automap, avg_mse_unrolled_base, avg_mse_unrolled_vgg];
runtime_arr_ms = [reconTime_admm_ms, reconTime_fista_ms, reconTime_fc_ms, reconTime_automap_ms, reconTime_unrolled_base_ms, reconTime_unrolled_vgg_ms];

savedir = fullfile(resultpath, 'exp');
final_report_str = "";
for i = 1:length(label_arr)
    str_line = sprintf("%s reconstruction. SSIM: %.2f; MSE: %.2d; Runtime: %.2d\n", label_arr(i), ssim_arr(i), mse_arr(i), runtime_arr_ms(i));
    fprintf(str_line);
    final_report_str = sprintf("%s%s", final_report_str, str_line);
end

report_filename = "avg_mse_runtime.txt";
fid = fopen(fullfile(resultpath, 'exp', report_filename), 'w');
fprintf(fid, final_report_str);





