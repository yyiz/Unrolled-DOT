clear; close all; clc;

addpath("../setpaths");
addpath(genpath("../"));
[libpath, datpath, resultpath, basepath] = setpaths;
addpath(sprintf("%s/export_fig", libpath));
addpath(sprintf("%s/subaxis", libpath));

vis_inds = [
    {4,2,3,3};
    {2,2,2,2};
    {4,4,3,4};
];
edge_colors = ['r'; 'c'; 'm'];

nInds = size(vis_inds,1);

pad = 0.005;
spacing_config1 = {'SpacingHoriz', 0.0, 'SpacingVert', 0.0,...
        'PaddingTop', pad, 'PaddingBottom', pad, 'PaddingLeft', pad, 'PaddingRight', pad,...
        'margin', 0.00};

%% Prepare save dir

savedir = sprintf("%s/jacobian_compare/", resultpath);
if ~exist(savedir, 'dir')
    mkdir(savedir);
end
savename = "jacobian_compare";
fullpath = sprintf("%s/%s", savedir, savename);
    
%% Load MC Jacobian

datname = "allTrainingDat_30-Sep-2021";
processed_dat_EML = sprintf("%s_EML", datname);
procDat = load(sprintf("%s/%s", datpath, processed_dat_EML));

loadStruct(procDat.Jheaders);

Jsim = procDat.J_L;
Jsim_reshape = reshape(Jsim, SRC_W, SRC_L, SENS_W, SENS_L, VOX_W, VOX_L);

%% Load learned Jacobian

modelname = "model_allTrainingDat_30-Sep-2021_EML_NL=1_nEpoch=2000_lossFunc=MAE_untied=T_vgg=F_unet_nfilts=0_act=shrink";
model_path = sprintf("%s/exp/%s.pt", resultpath, modelname);


run_runtime_cmd_base = sprintf("python3 get_W.py %s", model_path);
[~, W_savepath] = system(run_runtime_cmd_base);
if startsWith(W_savepath, "'python3' is not recognized")
    run_runtime_cmd_base = sprintf("python get_W.py %s", model_path);
    [~, W_savepath] = system(run_runtime_cmd_base);
end

W_struct = load(strip(W_savepath));

WT = W_struct.WT;

WT_reshape = reshape(WT, SRC_W, SRC_L, SENS_W, SENS_L, VOX_W, VOX_L);


%% Visualize results

f = figure('Position', [575 665 1125 190]); 
for i = 1:nInds
vis_inds_i = vis_inds(i,:);
    
J_i = squeeze(Jsim_reshape(vis_inds_i{:},:,:));
WT_i = rot90(squeeze(WT_reshape(vis_inds_i{:},:,:)),1)';

subaxis(1,2*nInds,i, spacing_config1{:});
imagesc(J_i); set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]); axis image;
rectangle('Position',[1,1,40,40], 'LineWidth', 3, 'EdgeColor', edge_colors(i))

subaxis(1,2*nInds,i+nInds, spacing_config1{:});
imagesc(WT_i); set(gca,'XTickLabel',[]); set(gca,'YTickLabel',[]); axis image;
rectangle('Position',[1,1,40,40], 'LineWidth', 3, 'EdgeColor', edge_colors(i))
end

%% Save results

export_fig(f, fullpath, '-m3', '-png', '-transparent');


