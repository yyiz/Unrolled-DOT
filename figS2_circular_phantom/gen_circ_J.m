clear; close all; clc;

addpath("../setpaths");
[libpath, datpath, resultpath, basepath] = setpaths;

nAngs = 32;
depth_slices = 35;
nbins_gate = 64;
nvox_final = 16;
nbins_final = 4;

Jdir = '11_12_22_circular_jac';
Jname = 'J';
tpsfdir = Jdir;
tpsfname = 'tpsf0';

load(sprintf("%s/%s/%s.mat", datpath, Jdir, Jname));
load(sprintf("%s/%s/%s.mat", datpath, tpsfdir, tpsfname));

J_reshape = reshape(J, [Jheaders.NBINS, Jheaders.VOX_W, Jheaders.VOX_L, Jheaders.VOX_H]);

Jheaders.VOX_L = nvox_final;
Jheaders.VOX_W = nvox_final;
Jheaders.VOX_H = nvox_final;


r = Jheaders.VOX_W/2;
x_vec = linspace(-Jheaders.VOX_W/2+1,Jheaders.VOX_W/2,Jheaders.VOX_W);
y_vec = linspace(-Jheaders.VOX_L/2+1,Jheaders.VOX_L/2,Jheaders.VOX_L);
[x_grid, y_grid] = meshgrid(x_vec, y_vec);
in_circ = x_grid.^2 + y_grid.^2 <= r.^2;

angs_vec = linspace(0, 180, nAngs+1);
angs_vec = angs_vec(1:(end-1));

J_gated = squeeze(sum(J_reshape(1:nbins_gate,:,depth_slices,:), 3));
bkg_gated = bkgTpsf(1:nbins_gate);

J_slice = imresize(permute(J_gated, [2,3,1]), [nvox_final, nvox_final]);

for a_i = 1:nAngs
    
    rot_i = angs_vec(a_i);
    J_slice_i = imrotate(J_slice, rot_i, 'bilinear', 'crop') .* in_circ;

    if a_i == 1
        J_full = J_slice_i;
    else
        J_full = cat(4, J_full, J_slice_i);
    end
end

% Further bin measurements
bin_sz = ceil(length(1:nbins_gate)./nbins_final);
J_binned = zeros(nvox_final, nvox_final, nbins_final, nAngs);
bkg_binned = zeros(nbins_final,1);
for T = 1:nbins_final
    ind1 = (T-1)*bin_sz + 1;
    ind2 = min(T*bin_sz, size(J_full,3));
    J_binned(:,:,T,:) = sum(J_full(:,:,ind1:ind2,:), 3);
    bkg_binned(T) = sum(bkg_gated(ind1:ind2));
end

Jheaders.NBINS = nbins_final;
J_final = reshape(J_binned, [Jheaders.VOX_L*Jheaders.VOX_H, Jheaders.NBINS*nAngs])';
bkg_final = reshape(repmat(bkg_binned, [1,nAngs]), [Jheaders.NBINS*nAngs,1]);

savefname = sprintf('%s/%s/J_multisrc_interp.mat', datpath, Jdir);
save(savefname, '-v7.3', 'J_final', 'bkg_final', 'Jheaders')

