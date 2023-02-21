from parse import *

def setpaths(paths_dir):
    with open('%s/paths.txt' % paths_dir, 'r') as f:
        all_lines = f.readlines()
    libpath = parse('libpath={}', "".join(all_lines[0].split()))
    datpath = parse('datpath={}', "".join(all_lines[1].split()))
    resultpath = parse('resultpath={}', "".join(all_lines[2].split()))
    basepath = parse('basepath={}', "".join(all_lines[3].split()))
    return libpath[0], datpath[0], resultpath[0], basepath[0]
