clear; close all; clc;

rng(1);

addpath("../setpaths");
[libpath, datpath, resultpath, basepath] = setpaths;

addpath(genpath(libpath));

ncols = 4;

% Load in end-to-end training data
end2end_fname = "model_allTrainingDat_30-Sep-2021_EML_NL=1_nEpoch=400_lossFunc=MAE_untied=T_vgg=T_unet_nfilts=16_act=shrink";
end2end_struct = load(sprintf("%s/exp/%s", resultpath, end2end_fname));

% Load in 2-step training data
two_step_fname = "model_2-step_pretrained=model_allTrainingDat_30-Sep-2021_EML_NL=1_nEpoch=2000_lossFunc=MAE_untied=T_vgg=F_unet_nfilts=0_act=shrink";
two_step_struct = load(sprintf("%s/exp/%s", resultpath, two_step_fname));

spacing_config1 = {'SpacingHoriz', 0.0, 'SpacingVert', 0.0,...
        'PaddingTop', 0.00, 'PaddingLeft', 0.00, 'margin', 0.00};

im_sz = size(end2end_struct.reconIms_np, [1,2]);
showInds = randi(size(end2end_struct.reconIms_np, 3), ncols, 1);

%%
savedir = sprintf("%s/two-step", resultpath);
truthIms = permute(reshape(end2end_struct.truthIms_np, im_sz(1), im_sz(2), []), [2,1,3]);
metrics_and_plot(truthIms, end2end_struct.reconIms_np, two_step_struct.reconIms_np, "End-to-end", "2-Step", showInds, savedir);


