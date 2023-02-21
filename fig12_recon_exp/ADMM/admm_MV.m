function [x, errVec] = admm_MV(A, b, Dx, Dy, opts)

AT = transpose(A);

szX = size(A,2);
x = zeros(szX,1);

eta_nneg = opts.eta_nneg_init.*ones(size(x));
eta_L1 = opts.eta_L1_init.*ones(size(x));

fidFn = @(X) sum(reshape(A*X - b,[],1).^2) + opts.lamL2*sum(X(:).^2);
errVec = zeros(opts.maxIters, 1);

ATy = AT*b;
ATA_tik = AT*A + opts.lamL2*speye(szX);

Dx_adj = transpose(Dx);
DxTDx = Dx_adj*Dx;

Dy_adj = transpose(Dy);
DyTDy = Dy_adj*Dy;

eta_TV_dx = opts.eta_TV_init.*ones(size(Dx,1),1);
eta_TV_dy = opts.eta_TV_init.*ones(size(Dy,1),1);

for i = 1:opts.maxIters
    
    errVec(i) = fidFn(x);
    
    x_prv = x;

    % Primal updates
    w = max(0, x + eta_nneg ./ opts.mu_nneg);
    z_dx = shrinkageOp(Dx*x + eta_TV_dx./opts.mu_TV, opts.gam_TV./opts.mu_TV);
    z_dy = shrinkageOp(Dy*x + eta_TV_dy./opts.mu_TV, opts.gam_TV./opts.mu_TV);
    q = shrinkageOp(x + eta_L1/opts.mu_L1, opts.gam_L1/opts.mu_L1);
    x = (ATA_tik + opts.mu_nneg*speye(szX) + opts.mu_L1*speye(szX) + opts.mu_TV*DxTDx + opts.mu_TV*DyTDy)\...
        (ATy + Dx_adj*(opts.mu_TV.*z_dx - eta_TV_dx) +Dy_adj*(opts.mu_TV*z_dy-eta_TV_dy) + (opts.mu_L1*q - eta_L1) + (opts.mu_nneg*w - eta_nneg));    

    % Update primary residuals
    res_z_dx_prim = Dx*x - z_dx;
    res_z_dy_prim = Dy*x - z_dy;
    res_q_prim = x - q;
    res_w_prim = x - w;
    
    % Update dual variables
    eta_TV_dx = eta_TV_dx + opts.mu_TV.*res_z_dx_prim;
    eta_TV_dy = eta_TV_dy + opts.mu_TV.*res_z_dy_prim;
    eta_L1 = eta_L1 + opts.mu_L1.*res_q_prim;
    eta_nneg = eta_nneg + opts.mu_nneg.*res_w_prim;
    
    % Update the dual residuals
    res_z_dx_dual = Dx*(x - x_prv);
    res_z_dy_dual = Dy*(x - x_prv);
    res_q_dual = x - x_prv;
    res_w_dual = x - x_prv;
    
    % Update step sizes
    if norm(res_w_prim) > opts.eps*norm(res_w_dual)
        opts.mu_nneg = opts.tau_inc * opts.mu_nneg;
    elseif norm(res_w_dual) > opts.eps*norm(res_w_prim)
        opts.mu_nneg = opts.mu_nneg ./ opts.tau_dec;
    else
        % opts.mu_nneg = opts.mu_nneg;
    end
    
    if norm(res_q_prim) > opts.eps*norm(res_q_dual)
        opts.mu_L1 = opts.tau_inc * opts.mu_L1;
    elseif norm(res_q_dual) > opts.eps*norm(res_q_prim)
        opts.mu_L1 = opts.mu_L1 ./ opts.tau_dec;
    else
        % opts.mu_L1 = opts.mu_L1;
    end

    res_z_prim = sqrt((norm(res_z_dy_prim)).^2 + (norm(res_z_dx_prim)).^2);
    res_z_dual = sqrt((norm(res_z_dy_dual)).^2 + (norm(res_z_dx_dual)).^2);
    if res_z_prim > opts.eps*res_z_dual
        opts.mu_TV = opts.tau_inc * opts.mu_TV;
    elseif res_z_dual > opts.eps*res_z_prim
        opts.mu_TV = opts.mu_TV ./ opts.tau_dec;
    else
        % opts.mu_TV = opts.mu_TV;
    end
%     resz_prim = norm(res_z_dx_prim);
%     resz_dual = norm(res_z_dx_dual);
%     if resz_prim > opts.eps*resz_dual
%         opts.mu_TV = opts.tau_inc * opts.mu_TV;
%     elseif resz_dual > opts.eps*resz_prim
%         opts.mu_TV = opts.mu_TV ./ opts.tau_dec;
%     else
%         % opts.mu_TV = opts.mu_TV;
%     end
end

end

function s = shrinkageOp(x,lmbd)
%% Author:
% Vivek Boominathan
% Rice University
% vivekb@rice.edu

%%
s = max(abs(x)-lmbd,0).*sign(x);
end
