clear; close all; clc;

addpath("../setpaths");
[libpath, datpath, resultpath, basepath] = setpaths;

addpath(genpath(libpath));

nLayers_arr = [1,3,5,7,10];

lossFunc = 'MAE';
plot_font = 'Times New Roman';
plot_fontsize = 18;
interpOrder = 2;
ninterp = 8;

tests_type = {"Simulation", "Experimental"};
nPlots = length(tests_type);

result_dir = sprintf("%s/sim_exp_nlayers", resultpath);
if ~exist(result_dir, 'dir')
    mkdir(result_dir);
end

for i = 1:nPlots

    test_i = tests_type{i};

    nLayers = length(nLayers_arr);
    err_vec = zeros(nLayers, 1);
    runtime_vec_NL = zeros(nLayers,1);
    for NL_i = 1:length(nLayers_arr)


        if strcmp(test_i, "Experimental")
            modelname_i = sprintf("result_model_allTrainingDat_30-Sep-2021_EML_NL=%d_nEpoch=2000_lossFunc=MAE_untied=T_actfunc=shrink.mat", nLayers_arr(NL_i));
            load(sprintf("%s/performance_analysis/%s", resultpath, modelname_i));
        else
            modelname_i = sprintf("model_5_1_22_unrolled_jac_train=m_test=m_NL=%d_nEpoch=2000_lossFunc=MAE_untied=T_vgg=F_unet_nfilts=0.mat", nLayers_arr(NL_i));
            load(sprintf("%s/sim/test_nlayers/%s", resultpath, modelname_i));
        end
        
        err_vec(NL_i) = test_losses(end);
        runtime_vec_NL(NL_i) = sum(runtime_arr(:));
    end
    
    runtime_vec_NL_min = runtime_vec_NL ./ 60; % convert from seconds to minutes
    
    f = figure('Position', [680 280 560 600]);
    set(gca,'FontSize', plot_fontsize, 'FontName', plot_font);
    
    p_loss = polyfit(nLayers_arr, err_vec', interpOrder);
    interpAx_loss = linspace(nLayers_arr(1), nLayers_arr(end), ninterp);
    interpP_loss = polyval(p_loss, interpAx_loss);
    
    p_time = polyfit(nLayers_arr, runtime_vec_NL_min', interpOrder);
    interpAx_time = linspace(nLayers_arr(1), nLayers_arr(end), ninterp);
    interpP_time = polyval(p_time, interpAx_time);

    yyaxis left;
    scatter(nLayers_arr, err_vec, 50, 'filled');
    hold on;
    plot(interpAx_loss, interpP_loss, '--', 'LineWidth', 2);
    xlabel("Number of Layers");
    ylabel(lossFunc);

    yyaxis right;
    scatter(nLayers_arr, runtime_vec_NL_min, 50, 'filled');
    hold on;
    plot(interpAx_time, interpP_time);
    xlabel("Number of Layers");
    ylabel("Runtime (min)");

    lgd = legend(sprintf("%s Loss", test_i), "Interpolation (Loss)", sprintf("%s Runtime", test_i), "Interpolation", "Location", "northeast");
    lgd.Position = lgd.Position + [-0.05, +0.05, 0, 0];

    export_fig(f, sprintf("%s/%s_test_nlayers", result_dir, test_i), '-png', '-m3', '-transparent');
end

