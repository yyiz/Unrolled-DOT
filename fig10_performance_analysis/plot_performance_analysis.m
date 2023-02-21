clear; close all; clc;

addpath("../setpaths");
[libpath, datpath, resultpath, basepath] = setpaths;

addpath(genpath(libpath));

untied_str = "T";
datfilename = "allTrainingDat_30-Sep-2021_EML";

actfunc = "shrink";
lossFunc = "MAE";
nitr = 2000;
NL = 3;

plot_font = 'Times New Roman';
plot_fontsize = 18;

showInd = 3;

%% Test number of iterations

NL_vec = [250, 500, 2000];

for k = 1:length(NL_vec)
    NL_k = NL_vec(k);
    fname_itrs = sprintf("result_model_%s_NL=%d_nEpoch=%d_lossFunc=%s_untied=%s_actfunc=%s", datfilename, NL, NL_k, lossFunc, untied_str, actfunc);
    fullpath_itrs = sprintf("%s/performance_analysis/%s", resultpath, fname_itrs);
    load(fullpath_itrs);
    
    showIm = reconIms(:,:,showInd);
    showIm = max(showIm, 0);
    
    f = figure(); imagesc(showIm); set(gca, 'XTickLabels', []); set(gca, 'YTickLabels', []); axis image;
    title(num2str(NL_k));
    set(gca,'FontSize', 40, 'FontName', plot_font);
    export_fig(f, sprintf("%s/performance_analysis/testItrs_reconIm_nitr=%d", resultpath, NL_k), '-m3', '-png', '-transparent');
end

f = figure('Position', [680 280 560 600]); 
semilogy(epoch_arr, train_losses, 'LineWidth', 2);
hold on
semilogy(epoch_arr, test_losses, 'LineWidth', 2);
legend("Training Loss", "Test Loss", "Location", "northeast");
xlabel("Number of Training Iterations");
ylabel(lossFunc);
set(gca,'FontSize', plot_fontsize, 'FontName', plot_font);
export_fig(f, sprintf("%s/performance_analysis/testItrs_err_falloff", resultpath), '-m3', '-png', '-transparent');

%% Test loss function

lossFunc_vec = ["SSIM", "MSE", "MAE"];
bw = 0.4;
mse_lims = [0, 4e-2];
bar_color = [61, 38, 168];

losses_mse_vec = zeros(length(lossFunc_vec),1);

for j = 1:length(lossFunc_vec)
    loss_j = lossFunc_vec(j);
    fname_loss = sprintf("result_model_%s_NL=%d_nEpoch=%d_lossFunc=%s_untied=%s_actfunc=%s", datfilename, NL, nitr, loss_j, untied_str, actfunc);
    fullpath_loss = sprintf("%s/performance_analysis/%s", resultpath, fname_loss);
    load(fullpath_loss);
    
    reconIms_norm = max(reconIms ./ max(reconIms, [], [1,2]), 0);
    truthIms_norm = max(truthIms ./ max(truthIms, [], [1,2]), 0);

    showIm = reconIms(:,:,showInd);
    showIm = max(showIm, 0);
    
    f = figure(); imagesc(showIm); set(gca, 'XTickLabels', []); set(gca, 'YTickLabels', []); axis image;
    title(loss_j);
    set(gca,'FontSize', 40, 'FontName', plot_font);
    export_fig(f, sprintf("%s/performance_analysis/testLoss_loss=%s", resultpath, loss_j), '-m3', '-png', '-transparent');
    
    avg_loss_j = 0;
    nTests = size(reconIms_norm, 3);
    for i = 1:nTests
        avg_loss_j = avg_loss_j + 2*mse(reconIms_norm(:,:,i), truthIms_norm(:,:,i));
    end
    avg_loss_j = avg_loss_j / nTests;
    losses_mse_vec(j) = avg_loss_j;
end

label_pos = 1:length(lossFunc_vec);

f = figure('Position', [680 280 560 600]);
bar(label_pos, losses_mse_vec, 'FaceColor', bar_color./255, 'BarWidth', bw);
set(gca,'FontSize', plot_fontsize, 'FontName', plot_font);
xlim([label_pos(1)-0.5, label_pos(end)+0.5]);
xticks(label_pos);
xticklabels(lossFunc_vec);
xlabel("Objective Function for Training Model");
ylabel("Reconstruction Quality of Test Set (MSE)");
ylim(mse_lims);
% title("Loss Function");

offset = 0.15;
for i1=1:length(losses_mse_vec)
    labPos = label_pos(i1);
    text(labPos,losses_mse_vec(i1),sprintf("%.2d", losses_mse_vec(i1)),'FontName', plot_font,...
        'FontSize', plot_fontsize,...
        'HorizontalAlignment','center',...
        'VerticalAlignment','bottom')
end

export_fig(f, sprintf("%s/performance_analysis/testLoss_mse_comparison", resultpath), '-m3', '-png', '-transparent');

%% Test number of layers

NL_vec = [1,3,5,7,10];
MAE_vec_NL = zeros(length(NL_vec),1);

for k = 1:length(NL_vec)
    NL_k = NL_vec(k);
    fname_NL = sprintf("result_model_%s_NL=%d_nEpoch=%d_lossFunc=%s_untied=%s_actfunc=%s", datfilename, NL_k, nitr, lossFunc, untied_str, actfunc);
    fullpath_NL = sprintf("%s/performance_analysis/%s", resultpath, fname_NL);
    load(fullpath_NL);
    MAE_vec_NL(k) = test_losses(end);
    
    showIm = reconIms(:,:,showInd);
    showIm = max(showIm, 0);
    
    f = figure(); imagesc(showIm); set(gca, 'XTickLabels', []); set(gca, 'YTickLabels', []); axis image;
    title(sprintf("%d Layers", NL_k));
    set(gca,'FontSize', 50, 'FontName', plot_font);
    export_fig(f, sprintf("%s/performance_analysis/testNL_reconIm_nitr=%d", resultpath, NL_k), '-m3', '-png', '-transparent');
end

f = figure('Position', [680 280 560 600]);

interpOrder = 3;
p = polyfit(NL_vec, MAE_vec_NL', interpOrder);

ninterp = 8;
interpAx = linspace(NL_vec(1), NL_vec(end), ninterp);
interpP = polyval(p, interpAx);

scatter(NL_vec, MAE_vec_NL, 50, 'filled');
hold on;
plot(interpAx, interpP, '--', 'LineWidth', 2);
xlabel("Number of Layers");
ylabel(lossFunc);
legend("Experimental Result", "Interpolation", "Location", "northeast");
set(gca,'FontSize', plot_fontsize, 'FontName', plot_font);
export_fig(f, sprintf("%s/performance_analysis/testNL_err", resultpath), '-m3', '-png', '-transparent');

%% Save truth Image

f = figure();
imagesc(truthIms(:,:,showInd)); set(gca, 'XTickLabels', []); set(gca, 'YTickLabels', []); axis image;
title("Truth");
set(gca,'FontSize', 40, 'FontName', plot_font);
export_fig(f, sprintf("%s/performance_analysis/truth", resultpath), '-m3', '-png', '-transparent');



