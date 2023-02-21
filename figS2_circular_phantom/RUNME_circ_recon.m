clear; close all; clc;

addpath("../setpaths");
[libpath, datpath, resultpath, basepath] = setpaths;

addpath(sprintf("%s/fig12_recon_exp/benchmark_linear_solvers", basepath));
addpath(sprintf("%s/fig12_recon_exp/support_scripts", basepath));
addpath(genpath(libpath));
addpath(genpath(sprintf("%s/lib", basepath)));

Jac_file = 'model_11_12_22_circular_jac_train=m_test=m_NL=3_nEpoch=6000_lossFunc=MAE_untied=T_vgg=F_unet_nfilts=0.mat';

simpath = sprintf("%s/sim", resultpath);

J_struct = load(sprintf("%s/%s", simpath, Jac_file));
testInds = 1:size(J_struct.truthIms,3);

%% 1. Preprocess experimentally collected data


%% 2. Generate FISTA results for comparison

fistaOpts.lam1 = 5e-2;
fistaOpts.lam2 = 0;
fistaOpts.maxItr = 50;
fistaOpts.tol = 0;
fistaOpts.nonneg = true; 
fistaOpts.showFigs = false;
fistaOpts.avgMaxK = 5; 
fistaOpts.scaleMag_J = 1;
fistaOpts.scaleMag_m = 1;
fistaOpts.shouldSave = false;

[reconIms_fista, reconTime_fista] = reconFISTA_exp(simpath, resultpath, Jac_file, testInds, fistaOpts,...
    "J_mat_np", "diff_meas");


%% Compute metrics

savedir = sprintf("%s/circ", resultpath);
plotInds = [15, 19, 24, 28];
metrics_and_plot(J_struct.truthIms, reconIms_fista, J_struct.recon_test_np, "FISTA", "Unrolled", plotInds, savedir);







