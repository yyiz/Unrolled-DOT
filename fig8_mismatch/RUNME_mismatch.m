clear; close all; clc;

addpath("../setpaths");
[libpath, datpath, resultpath, basepath] = setpaths;

addpath(genpath(libpath));

nitrs_vec = [0, 200, 500, 1999];
display_ims = [15, 112, 18, 290];
displayJac_ims = [
    {4,2,3,3};
    {3,3,3,3};
    {4,4,3,4};
    {2,4,2,4};
];

det_SL = 3;

im_shape = [41,41];
srcdet_shape = [5,5,5,5];

dat_fname = "model_allTrainingDat_30-Sep-2021_EML_mis_NL=1_nEpoch=2000_lossFunc=MAE_untied=T_vgg=F_unet_nfilts=0_act=shrink";
true_jac_path = sprintf("%s/allTrainingDat_30-Sep-2021_EML.mat", datpath);
loadpath = sprintf("%s/exp/intermed_%s", resultpath, dat_fname);

spacing_config0 = {'SpacingHoriz', 0.0, 'SpacingVert', 0.0,...
        'PaddingTop', 0.00, 'PaddingBottom', 0.00, 'PaddingLeft', 0.00, 'margin', 0.00};
spacing_config1 = {'SpacingHoriz', 0.0, 'SpacingVert', 0.0,...
        'PaddingTop', 0.00, 'PaddingBottom', 0.01, 'PaddingLeft', 0.00, 'margin', 0.00};
spacing_config2 = {'SpacingHoriz', 0.0, 'SpacingVert', 0.0,...
        'PaddingTop', 0.01, 'PaddingBottom', 0.00, 'PaddingLeft', 0.00, 'margin', 0.00};

J_truth_struct = load(true_jac_path);
J_truth = reshape(J_truth_struct.J_L, [srcdet_shape, im_shape]);

dat_struct = load(sprintf("%s/exp/%s", resultpath, dat_fname));
all_truth_ims = reshape(dat_struct.truthIms_np, im_shape(1), im_shape(2), []);

VOX_L = J_truth_struct.Jheaders.VOX_L;
VOX_W = J_truth_struct.Jheaders.VOX_W;
SRC_L = J_truth_struct.Jheaders.SRC_L;
SRC_W = J_truth_struct.Jheaders.SRC_W;
SENS_L = J_truth_struct.Jheaders.SENS_L;
SENS_W = J_truth_struct.Jheaders.SENS_W;

f_jac = figure('Position', [360, 80, 350, 350]);
f_ims = figure('Position', [360, 80, 350, 350]);
for J_ind = 1:length(display_ims)
    inds_im_i = displayJac_ims(J_ind,:);
    
    J_truth_i = rot90(squeeze(J_truth(inds_im_i{:},:,:))', -1);
    truth_im_i = all_truth_ims(:,:,display_ims(J_ind))';

    figure(f_ims);
    if J_ind > 2
        subaxis(2,2,J_ind,spacing_config1{:});
    else
        subaxis(2,2,J_ind,spacing_config0{:});
    end
    imagesc(truth_im_i); set(gca, 'XTickLabels', []); set(gca, 'YTickLabels', []);
    set(gca,'XColor',[1 1 1]); set(gca,'YColor',[1 1 1]);

    figure(f_jac);
    if J_ind < 3
        subaxis(2,2,J_ind,spacing_config2{:});
    else
        subaxis(2,2,J_ind,spacing_config0{:});
    end
    imagesc(J_truth_i); set(gca, 'XTickLabels', []); set(gca, 'YTickLabels', []);
    set(gca,'XColor',[1 1 1]); set(gca,'YColor',[1 1 1]);

    src_pos_L = (VOX_L/(SRC_L*2)) + (inds_im_i{1}-1)*(VOX_L/SRC_L);
    src_pos_W = VOX_W - (VOX_W/(SRC_W*2)) - (inds_im_i{2}-1)*(VOX_W/SRC_W);
    sens_pos_L = (VOX_L/(SENS_L*2)) + (inds_im_i{3}-1)*(VOX_L/SENS_L);
    sens_pos_W = VOX_W - (VOX_W/(SENS_W*2)) - (inds_im_i{4}-1)*(VOX_W/SENS_W);
    hold on;
    rectangle('Position', [sens_pos_W-det_SL./2, sens_pos_L-det_SL./2, det_SL, det_SL], "FaceColor",'b', 'EdgeColor','w')
    scatter(src_pos_W, src_pos_L, 'filled', 'r', 'LineWidth', 1, 'MarkerEdgeColor', 'w');
end

f_im_vec = [];
f_jac_vec = [];
for ni = 1:length(nitrs_vec)
    load_fname = sprintf("%s/itr=%d", loadpath, nitrs_vec(ni));
    load(load_fname);
    
    reconIms_ni = reshape(recon_ims_test, im_shape(1), im_shape(2), []);
    J_ni = reshape(WT, [srcdet_shape, im_shape]);

    if nitrs_vec(ni) == 0
        recon_itr0 = reconIms_ni;
    elseif nitrs_vec(ni) == 1999
        recon_itr1999 = reconIms_ni;
    end
    
    f1 = figure('Position', [360, 80, 350, 350]);
    f2 = figure('Position', [360, 80, 350, 350]);

    for im_i = 1:length(display_ims)

        figure(f1);
        if im_i > 2
            subaxis(2,2,im_i,spacing_config1{:});
        else
            subaxis(2,2,im_i,spacing_config0{:});
        end
        imagesc(reconIms_ni(:,:,display_ims(im_i))'); set(gca, 'XTickLabels', []); set(gca, 'YTickLabels', []);
        set(gca,'XColor',[1 1 1]); set(gca,'YColor',[1 1 1]);

        figure(f2);
        inds_im_i = displayJac_ims(im_i,:);        
        if im_i < 3
            subaxis(2,2,im_i,spacing_config2{:});
        else
            subaxis(2,2,im_i,spacing_config0{:});
        end
        imagesc(squeeze(J_ni(inds_im_i{:},:,:))); set(gca, 'XTickLabels', []); set(gca, 'YTickLabels', []);
        set(gca,'XColor',[1 1 1]); set(gca,'YColor',[1 1 1]);

        src_pos_L = (VOX_L/(SRC_L*2)) + (inds_im_i{1}-1)*(VOX_L/SRC_L);
        src_pos_W = VOX_W - (VOX_W/(SRC_W*2)) - (inds_im_i{2}-1)*(VOX_W/SRC_W);
        sens_pos_L = (VOX_L/(SENS_L*2)) + (inds_im_i{3}-1)*(VOX_L/SENS_L);
        sens_pos_W = VOX_W - (VOX_W/(SENS_W*2)) - (inds_im_i{4}-1)*(VOX_W/SENS_W);
        hold on;
        rectangle('Position', [sens_pos_W-det_SL./2, sens_pos_L-det_SL./2, det_SL, det_SL], "FaceColor",'b', 'EdgeColor','w')
        scatter(src_pos_W, src_pos_L, 'filled', 'r', 'LineWidth', 1, 'MarkerEdgeColor', 'w');
    end
    f_im_vec = [f_im_vec, f1];
    f_jac_vec = [f_jac_vec, f2];
end


avg_mse_im1 = 0;
avg_corr_coeff_im1 = 0;
avg_mse_im2 = 0;
avg_corr_coeff_im2 = 0;
nTests = size(recon_itr0,3);
for k = 1:nTests
    avg_mse_im1 = avg_mse_im1 + 2*mse(recon_itr0(:,:,k), all_truth_ims(:,:,k));
    corr_coeff_k = corrcoef(recon_itr0(:,:,k), all_truth_ims(:,:,k));
    avg_corr_coeff_im1 = avg_corr_coeff_im1 + corr_coeff_k(1,2);
    avg_mse_im2 = avg_mse_im2 + 2*mse(recon_itr1999(:,:,k), all_truth_ims(:,:,k));
    corr_coeff_k = corrcoef(recon_itr1999(:,:,k), all_truth_ims(:,:,k));
    avg_corr_coeff_im2 = avg_corr_coeff_im2 + corr_coeff_k(1,2);
end
avg_mse_im1 = avg_mse_im1 / nTests;
avg_corr_coeff_im1 = avg_corr_coeff_im1 / nTests;
avg_mse_im2 = avg_mse_im2 / nTests;
avg_corr_coeff_im2 = avg_corr_coeff_im2 / nTests;

result_dir = sprintf("%s/model_mismatch", resultpath);
if ~exist(result_dir, 'dir')
    mkdir(result_dir);
end

metrics_savepath = sprintf("%s/metrics.txt", result_dir);
fid = fopen(metrics_savepath, 'w');
savestr = sprintf("Iteration 0. MSE: %f; Corr Coeff: %f\nIteration 1999. MSE: %f; Corr Coeff: %f\n", avg_mse_im1, avg_corr_coeff_im1, avg_mse_im2, avg_corr_coeff_im2);
fprintf(fid, savestr);
fprintf(savestr);

export_fig(f_ims, sprintf("%s/recon_truth", result_dir), '-png', '-m3', '-transparent');
export_fig(f_jac, sprintf("%s/J_truth", result_dir), '-png', '-m3', '-transparent');
for k = 1:length(f_jac_vec)
    export_fig(f_jac_vec(k), sprintf("%s/fig_jac_itr=%d", result_dir, nitrs_vec(k)), '-png', '-m3', '-transparent');
end
for k = 1:length(f_im_vec)
    export_fig(f_im_vec(k), sprintf("%s/fig_recon_itr=%d", result_dir, nitrs_vec(k)), '-png', '-m3', '-transparent');
end

