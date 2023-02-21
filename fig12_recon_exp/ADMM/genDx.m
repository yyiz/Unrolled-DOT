function [Dx, Dy] = genDx(Ix, Iy, Cx, Cy)

    DxPsf = fspecial('sobel');
    DyPsf = DxPsf';

    Dy_fd = Fx(DyPsf, Cy, Cx);
    Dyfunc = @(X) FiltX(Dy_fd, X, Cy, Cx);

    Dx_fd = Fx(DxPsf, Cy, Cx);
    Dxfunc = @(X) FiltX(Dx_fd, X, Cy, Cx);
    
    outSz = Cx*Cy;
    inSz = Ix*Iy;
    
    Dx = zeros(outSz, inSz);
    for i = 1:inSz
        imI = zeros(inSz,1);
        imI(i) = 1;
        imI = reshape(imI, Iy, Ix);
        imI_response = Dxfunc(imI);
        Dx(:,i) = imI_response(:);
    end
    
    Dy = zeros(outSz, inSz);
    for i = 1:inSz
        imI = zeros(inSz,1);
        imI(i) = 1;
        imI = reshape(imI, Iy, Ix);
        imI_response = Dyfunc(imI);
        Dy(:,i) = imI_response(:);
    end
end

%% Filter functions
function F = Fx(x, m, n)
    F = fft2(fftshift(x), m, n);
end

function y = FiltX(H, x, m, n)
    y = real(ifftshift(ifft2(H.*Fx(x, m, n), m, n)));
end