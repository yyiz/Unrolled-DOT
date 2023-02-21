function preprocessDatExp(datpath, datname, s, savename)

%% Prepare Jacobian

mcdir = "5_29_21_src-det_10x10_scene_4cm";

srcRowInds = 1:2:9;
srcColInds = 1:2:9;
detRowInds = 2:2:10;
detColInds = 2:2:10;

srcRowInds_mis = 2:6;
srcColInds_mis = 2:6;
detRowInds_mis = 2:6;
detColInds_mis = 2:6;

expdat_struct = load(sprintf("%s/%s", datpath, datname));
load(sprintf("%s/%s/J", datpath, mcdir));
loadStruct(Jheaders);
J_reshape = reshape(J, NBINS, SRC_W, SRC_L, SENS_W, SENS_L, VOX_W, VOX_L, VOX_H);
J_mis_temp = J_reshape;
J_reshape = J_reshape(:,srcColInds,srcRowInds,detColInds,detRowInds,:,:,:);

nSrcDet = prod([length(srcRowInds), length(srcColInds), length(detRowInds), length(detColInds)]);
nVox = prod([VOX_W, VOX_L, VOX_H]);

DTIME = TIME_MAX ./ NBINS;
timeAx = (DTIME./expdat_struct.bin_fac).*(1:NBINS)';
J_E = reshape(sum(J_reshape,1), nSrcDet, nVox);
J_M = reshape(trapz(timeAx,(timeAx.*J_reshape)), nSrcDet, nVox);
J_L = reshape(trapz(timeAx,(exp(-s*timeAx).*J_reshape)), nSrcDet, nVox);
J_EML = cat(1, J_E, J_M, J_L);

nSrcDet_mis = prod([length(srcRowInds_mis), length(srcColInds_mis), length(detRowInds_mis), length(detColInds_mis)]);

J_mis_temp = J_mis_temp(:,srcColInds_mis,srcRowInds_mis,detColInds_mis,detRowInds_mis,:,:,:);
% Testing model mismatch is only case in which Jacobian is provided as
% input for experimental data; therefore this is the only case where we
% need to make sure rotation and transpose match
J_mis = permute(rot90(permute(J_mis_temp, [7, 6, 1, 2, 3, 4, 5]), -1), [3, 4, 5, 6, 7, 1, 2]);

J_E_mis = reshape(sum(J_mis,1), nSrcDet_mis, nVox);
J_M_mis = reshape(trapz(timeAx,(timeAx.*J_mis)), nSrcDet_mis, nVox);
J_L_mis = reshape(trapz(timeAx,(exp(-s*timeAx).*J_mis)), nSrcDet_mis, nVox);
J_EML_mis = cat(1, J_E_mis, J_M_mis, J_L_mis);

%% Process experimental data

loadStruct(expdat_struct);
batchSz = 200;
nIms = size(allBkgDat,3);
nSrcDet = size(allBkgDat,2);
nBatches = ceil(nIms / batchSz);

nbins = size(allBkgDat,1);
timeAx = (1:nbins)';

bkg_E = sum(allBkgDat, 1);
abs_E = sum(allMeasDat,1);
bkg_M = zeros([1, nSrcDet, nIms]);
abs_M = zeros([1, nSrcDet, nIms]);
bkg_L = zeros([1, nSrcDet, nIms]);
abs_L = zeros([1, nSrcDet, nIms]);

for i = 1:nBatches
    start_i = (i-1)*batchSz + 1;
    end_i = min(i*batchSz, nIms);
    smoothed_bkg = smoothdata(allBkgDat(:,:,start_i:end_i), 1, 'gaussian', 100);
    smoothed_abs = smoothdata(allMeasDat(:,:,start_i:end_i),1, 'gaussian', 100);
    bkg_M(:,:,start_i:end_i) = trapz(timeAx,(timeAx.*smoothed_bkg));
    abs_M(:,:,start_i:end_i) = trapz(timeAx,(timeAx.*smoothed_abs));
    bkg_L(:,:,start_i:end_i) = trapz(timeAx,(exp(-s*timeAx).*smoothed_bkg));
    abs_L(:,:,start_i:end_i) = trapz(timeAx,(exp(-s*timeAx).*smoothed_abs));
end

diff_E = bkg_E - abs_E;
diff_M = bkg_M - abs_M;
diff_L = bkg_L - abs_L;

diff_E = diff_E ./ max(diff_E(:));
diff_M = diff_M ./ max(diff_M(:));
diff_L = diff_L ./ max(diff_L(:));

diff_EML = cat(1,diff_E, diff_M, diff_L);

truthSize = [41 41];
truthIms = imresize(truthIms, truthSize);

Jheaders.SRC_W = length(srcRowInds);
Jheaders.SRC_L = length(srcColInds);
Jheaders.SENS_W = length(detRowInds);
Jheaders.SENS_L = length(detColInds);

%% Save mismatched data

Jheaders_mis.SRC_W = length(srcRowInds_mis);
Jheaders_mis.SRC_L = length(srcColInds_mis);
Jheaders_mis.SENS_W = length(detRowInds_mis);
Jheaders_mis.SENS_L = length(detColInds_mis);

fullsavepath_mis = sprintf("%s/%s_mis", datpath, savename);
save(fullsavepath_mis, "diff_EML", "diff_E", "diff_M", "diff_L", "bkg_E", "abs_E", "bkg_M", "abs_M", "bkg_L", "abs_L",...
                   "J_E_mis", "J_M_mis", "J_L_mis", "J_EML_mis","Jheaders_mis",...
                   "truthIms");

%% Save confocal data

diff_EML_conf = zeros(3, Jheaders.SRC_W, Jheaders.SRC_L, nIms);
diff_E_conf = zeros(Jheaders.SRC_W, Jheaders.SRC_L, nIms);
diff_M_conf = zeros(Jheaders.SRC_W, Jheaders.SRC_L, nIms);
diff_L_conf = zeros(Jheaders.SRC_W, Jheaders.SRC_L, nIms);

bkg_E_conf = zeros(Jheaders.SRC_W, Jheaders.SRC_L, nIms);
bkg_M_conf = zeros(Jheaders.SRC_W, Jheaders.SRC_L, nIms);
bkg_L_conf = zeros(Jheaders.SRC_W, Jheaders.SRC_L, nIms);

abs_E_conf = zeros(Jheaders.SRC_W, Jheaders.SRC_L, nIms);
abs_M_conf = zeros(Jheaders.SRC_W, Jheaders.SRC_L, nIms);
abs_L_conf = zeros(Jheaders.SRC_W, Jheaders.SRC_L, nIms);

J_EML_conf = zeros(3, Jheaders.SRC_W, Jheaders.SRC_L, nVox);
J_E_conf = zeros(Jheaders.SRC_W, Jheaders.SRC_L, nVox);
J_M_conf = zeros(Jheaders.SRC_W, Jheaders.SRC_L, nVox);
J_L_conf = zeros(Jheaders.SRC_W, Jheaders.SRC_L, nVox);

diff_EML_temp = reshape(diff_EML, 3, Jheaders.SRC_W, Jheaders.SRC_L, Jheaders.SENS_W, Jheaders.SENS_L, nIms);
diff_E_temp = reshape(diff_E, Jheaders.SRC_W, Jheaders.SRC_L, Jheaders.SENS_W, Jheaders.SENS_L, nIms);
diff_M_temp = reshape(diff_M, Jheaders.SRC_W, Jheaders.SRC_L, Jheaders.SENS_W, Jheaders.SENS_L, nIms);
diff_L_temp = reshape(diff_L, Jheaders.SRC_W, Jheaders.SRC_L, Jheaders.SENS_W, Jheaders.SENS_L, nIms);

bkg_E_temp = reshape(bkg_E, Jheaders.SRC_W, Jheaders.SRC_L, Jheaders.SENS_W, Jheaders.SENS_L, nIms);
bkg_M_temp = reshape(bkg_M, Jheaders.SRC_W, Jheaders.SRC_L, Jheaders.SENS_W, Jheaders.SENS_L, nIms);
bkg_L_temp = reshape(bkg_L, Jheaders.SRC_W, Jheaders.SRC_L, Jheaders.SENS_W, Jheaders.SENS_L, nIms);

abs_E_temp = reshape(abs_E, Jheaders.SRC_W, Jheaders.SRC_L, Jheaders.SENS_W, Jheaders.SENS_L, nIms);
abs_M_temp = reshape(abs_M, Jheaders.SRC_W, Jheaders.SRC_L, Jheaders.SENS_W, Jheaders.SENS_L, nIms);
abs_L_temp = reshape(abs_L, Jheaders.SRC_W, Jheaders.SRC_L, Jheaders.SENS_W, Jheaders.SENS_L, nIms);

J_EML_temp = reshape(J_EML, 3, Jheaders.SRC_W, Jheaders.SRC_L, Jheaders.SENS_W, Jheaders.SENS_L, nVox);
J_E_temp = reshape(J_E, Jheaders.SRC_W, Jheaders.SRC_L, Jheaders.SENS_W, Jheaders.SENS_L, nVox);
J_M_temp = reshape(J_M, Jheaders.SRC_W, Jheaders.SRC_L, Jheaders.SENS_W, Jheaders.SENS_L, nVox);
J_L_temp = reshape(J_L, Jheaders.SRC_W, Jheaders.SRC_L, Jheaders.SENS_W, Jheaders.SENS_L, nVox);

for r = 1:Jheaders.SRC_L
    for c = 1:Jheaders.SRC_W
        diff_EML_conf(:, r, c, :) = diff_EML_temp(:, r, c, r, c, :);
        diff_E_conf(r, c, :) = diff_E_temp(r, c, r, c, :);
        diff_M_conf(r, c, :) = diff_M_temp(r, c, r, c, :);
        diff_L_conf(r, c, :) = diff_L_temp(r, c, r, c, :);
        
        bkg_E_conf(r, c, :) = bkg_E_temp(r, c, r, c, :);
        bkg_M_conf(r, c, :) = bkg_M_temp(r, c, r, c, :);
        bkg_L_conf(r, c, :) = bkg_L_temp(r, c, r, c, :);
        
        abs_E_conf(r, c, :) = abs_E_temp(r, c, r, c, :);
        abs_M_conf(r, c, :) = abs_M_temp(r, c, r, c, :);
        abs_L_conf(r, c, :) = abs_L_temp(r, c, r, c, :);
        
        J_EML_conf(:, r, c, :) =J_EML_temp(:, r, c, r, c, :);
        J_E_conf(r, c, :) = J_E_temp(r, c, r, c, :);
        J_M_conf(r, c, :) = J_M_temp(r, c, r, c, :);
        J_L_conf(r, c, :) = J_L_temp(r, c, r, c, :);
    end
end

fullsavepath_conf = sprintf("%s/%s_conf", datpath, savename);
save(fullsavepath_conf, "diff_EML_conf", "diff_E_conf", "diff_M_conf", "diff_L_conf",...
                   "bkg_E_conf", "abs_E_conf", "bkg_M_conf", "abs_M_conf", "bkg_L_conf", "abs_L_conf",...
                   "J_E_conf", "J_M_conf", "J_L_conf", "J_EML_conf",...
                   "Jheaders", "truthIms");

%% Save all results

fullsavepath = sprintf("%s/%s", datpath, savename);
save(fullsavepath, "diff_EML", "diff_E", "diff_M", "diff_L", "bkg_E", "abs_E", "bkg_M", "abs_M", "bkg_L", "abs_L",...
                   "J_E", "J_M", "J_L", "J_EML","Jheaders",...
                   "truthIms");


end
