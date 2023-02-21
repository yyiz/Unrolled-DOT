function metrics_and_plot(allTruth, im1, im2, name_im1, name_im2, showInds, savepath)
    truthIms = max(allTruth ./ max(allTruth, [], [1,2]), 0);
    reconIms_im1 = max(im1 ./ max(im1, [], [1,2]), 0);
    reconIms_im2 = max(im2 ./ max(im2, [], [1,2]), 0);
    
    nTests = size(reconIms_im2,3);
    avg_mse_im1 = 0;
    avg_corr_coeff_im1 = 0;
    avg_mse_im2 = 0;
    avg_corr_coeff_im2 = 0;
    for k = 1:nTests
        avg_mse_im1 = avg_mse_im1 + 2*mse(reconIms_im1(:,:,k), truthIms(:,:,k));
        corr_coeff_k = corrcoef(reconIms_im1(:,:,k), truthIms(:,:,k));
        avg_corr_coeff_im1 = avg_corr_coeff_im1 + corr_coeff_k(1,2);
        avg_mse_im2 = avg_mse_im2 + 2*mse(reconIms_im2(:,:,k), truthIms(:,:,k));
        corr_coeff_k = corrcoef(reconIms_im2(:,:,k), truthIms(:,:,k));
        avg_corr_coeff_im2 = avg_corr_coeff_im2 + corr_coeff_k(1,2);
    end
    avg_mse_im1 = avg_mse_im1 / nTests;
    avg_corr_coeff_im1 = avg_corr_coeff_im1 / nTests;
    avg_mse_im2 = avg_mse_im2 / nTests;
    avg_corr_coeff_im2 = avg_corr_coeff_im2 / nTests;
    
    if ~exist(savepath, 'dir')
        mkdir(savepath);
    end
    metrics_savepath = sprintf("%s/metrics.txt", savepath);
    fid = fopen(metrics_savepath, 'w');
    savestr = sprintf("%s. MSE: %f; Corr Coeff: %f\n%s. MSE: %f; Corr Coeff: %f\n", name_im1, avg_mse_im1, avg_corr_coeff_im1, name_im2, avg_mse_im2, avg_corr_coeff_im2);
    fprintf(fid, savestr);
    fprintf(savestr);
    
    %%
    ncols = length(showInds);
    nrows = 3;
    spacing_config1 = {'SpacingHoriz', 0.0, 'SpacingVert', 0.0,...
            'PaddingTop', 0.00, 'PaddingBottom', 0.01, 'PaddingLeft', 0.00, 'PaddingRight', 0.01, 'margin', 0.00};

    ims_savepath = sprintf("%s/saveims", savepath);
    box_w = 200;
    f_ims = figure('Position', [360, 140, ncols*box_w, nrows*box_w]); 
    for k = 1:ncols
        subaxis(nrows,ncols,k,spacing_config1{:});
        imagesc(reconIms_im1(:,:,showInds(k)));
        axis image; set(gca, 'XTickLabels', []); set(gca, 'YTickLabels', []);
        
        subaxis(nrows,ncols,k+ncols,spacing_config1{:});
        imagesc(reconIms_im2(:,:,showInds(k)));
        axis image; set(gca, 'XTickLabels', []); set(gca, 'YTickLabels', []);
        
        subaxis(nrows,ncols,k+2*ncols,spacing_config1{:});
        imagesc(truthIms(:,:,showInds(k)));
        axis image; set(gca, 'XTickLabels', []); set(gca, 'YTickLabels', []);
    end
    export_fig(f_ims, ims_savepath, '-m3', '-png', '-transparent');
end
