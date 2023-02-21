function [libpath, datpath, resultpath, basepath] = setpaths
    fid = fopen('paths.txt');
    libpath = sscanf(strrep(fgetl(fid), ' ', ''), "libpath=%s");
    datpath = sscanf(strrep(fgetl(fid), ' ', ''), "datpath=%s");
    resultpath = sscanf(strrep(fgetl(fid), ' ', ''), "resultpath=%s");
    basepath = sscanf(strrep(fgetl(fid), ' ', ''), "basepath=%s");
end