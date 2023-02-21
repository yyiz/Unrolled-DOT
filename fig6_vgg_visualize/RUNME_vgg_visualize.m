clear; close all; clc;

addpath("../setpaths");
[libpath, datpath, resultpath, basepath] = setpaths;

addpath(genpath(libpath));


vgg_fname = 'model_vgg_pretrained=model_allTrainingDat_30-Sep-2021_EML_NL=1_nEpoch=400_lossFunc=MAE_untied=T_vgg=T_unet_nfilts=16_act=shrink.mat';

full_loadpath = sprintf('%s/vgg/%s', resultpath, vgg_fname);
load(full_loadpath);

plot_inds = [10, 18, 41];


nplot_cols = 3;

spacing_config1 = {'SpacingHoriz', 0.0, 'SpacingVert', 0.0,...
        'PaddingTop', 0.00, 'PaddingBottom', 0.01, 'PaddingLeft', 0.00, 'margin', 0.00};
spacing_config2 = {'SpacingHoriz', 0.0, 'SpacingVert', 0.0,...
        'PaddingTop', 0.01, 'PaddingBottom', 0.00, 'PaddingLeft', 0.00, 'margin', 0.00};
spacing_config4 = {'SpacingHoriz', 0.0, 'SpacingVert', 0.0,...
        'PaddingTop', 0.01, 'PaddingBottom', 0.00, 'PaddingRight', 0.02, 'margin', 0.00};
spacing_config3 = {'SpacingHoriz', 0.0, 'SpacingVert', 0.0,...
        'PaddingTop', 0.00, 'PaddingBottom', 0.01, 'PaddingRight', 0.02, 'margin', 0.00};

vis_inds2_2 = 1:nplot_cols;
vis_inds4_3 = 1:nplot_cols;

for k = 1:length(plot_inds)
    f_truth = figure('Position', [680 430 350 700]);
    subaxis(2,1,1, spacing_config1{:});
    imagesc(truth_ims(:,:,plot_inds(k)));
    set(gca, "XTickLabels", []); set(gca, "YTickLabels", []);
    subaxis(2,1,2, spacing_config2{:});
    imagesc(recon_ims(:,:,plot_inds(k)));
    set(gca, "XTickLabels", []); set(gca, "YTickLabels", []);
    
    f_2_2 = figure('Position', [680 430 350*nplot_cols 700]);
    for j = 1:nplot_cols
        subaxis(2,nplot_cols,j, spacing_config3{:});
        imagesc(squeeze(vgg_truth2_2(plot_inds(k), vis_inds2_2(j), :, :)));
        set(gca, "XTickLabels", []); set(gca, "YTickLabels", []);
        subaxis(2,nplot_cols,j+nplot_cols, spacing_config4{:});
        imagesc(squeeze(vgg_recon2_2(plot_inds(k), vis_inds2_2(j), :, :)));
        set(gca, "XTickLabels", []); set(gca, "YTickLabels", []);
    end
    
    f_4_3 = figure('Position', [680 430 350*nplot_cols 700]);
    for j = 1:nplot_cols
        subaxis(2,nplot_cols,j, spacing_config3{:});
        imagesc(squeeze(vgg_truth4_3(plot_inds(k), vis_inds4_3(j), :, :)));
        set(gca, "XTickLabels", []); set(gca, "YTickLabels", []);
        subaxis(2,nplot_cols,j+nplot_cols, spacing_config4{:});
        imagesc(squeeze(vgg_recon4_3(plot_inds(k), vis_inds4_3(j), :, :)));
        set(gca, "XTickLabels", []); set(gca, "YTickLabels", []);
    end
    
    savedir = sprintf('%s/vgg/vgg_vis_ind=%d', resultpath, plot_inds(k));
    if ~exist(savedir, 'dir')
        mkdir(savedir);
    end
    savename_truth = sprintf('%s/truth_ind=%d', savedir, plot_inds(k));
    savename_2_2 = sprintf('%s/vgg2_2_ind=%d', savedir, plot_inds(k));
    savename_4_3 = sprintf('%s/vgg4_3_ind=%d', savedir, plot_inds(k));
    export_fig(savename_truth, f_truth, '-png', '-transparent', '-m3');
    export_fig(savename_2_2, f_2_2, '-png', '-transparent', '-m3');
    export_fig(savename_4_3, f_4_3, '-png', '-transparent', '-m3');
end

