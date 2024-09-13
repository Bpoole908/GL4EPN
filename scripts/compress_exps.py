
import os
import re
import zipfile
import tarfile
import argparse
import subprocess
from pathlib import Path
from pdb import set_trace
from tqdm import tqdm


def get_files(dirname, exclude=None):
    exclude = [] if exclude is None else exclude
    
    # setup file paths variable
    file_paths = []
    # Read all directory, subdirectories and file lists
    for root, dirs, files in os.walk(dirname):
        dirs[:] = [d for d in dirs if d not in exclude]
        for file_name in files:
            exclude_found =  [True for e in exclude if re.search(e, file_name)]
            if True in exclude_found:
                continue
            file_path = os.path.join(root, file_name)
            file_paths.append(file_path)
            
    return file_paths

def generate_name(name):
    return name.name[:-1] if str(name).endswith(os.path.sep) else name.name

def main(dirname, compress_name=None, exclude=None, dryrun=False):
    dirname = Path(dirname)
    
    og_dir = os.getcwd()
    os.chdir(dirname.parent)
    
    dirname = Path(dirname.name)
    compress_name = generate_name(dirname) if compress_name is None else compress_name
    compress_path = dirname.parent / Path(compress_name)
    if compress_path.suffix != '.tar.zst':
        compress_path = compress_path.with_suffix('.tar')
    print(f"Excluding the following files/dirs: {exclude}")
    # Call the function to retrieve all files and folders of the assigned directory
    file_paths = get_files(dirname, exclude=exclude)
    
    exclude = [f"--exclude={e}" for e in exclude]
    tar_cmd = ["tar", "-cvf", str(compress_path), *file_paths]
    # print(f"Tar command: {tar_cmd}")
    
    if dryrun:
        print('The following files will be compressed:')
        for f in file_paths:
            print(f)
        print(f"The compressed file will be save to '{os.path.join(os.getcwd(), compress_path)}'")
    else:
        code = subprocess.run(tar_cmd).returncode
        
        print(f"The directory '{dirname}' was successfully compressed to {os.path.join(os.getcwd(), compress_path)}")
        print(f"Compression size ~{os.path.getsize(compress_path) >> 20} MB")
        
    os.chdir(og_dir)
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train model")
    parser.add_argument('-d', '--dir', type=str, metavar='DIRECTORY', required=True,
                        help="Path to directory which will be compressed")
    parser.add_argument('-n', '--name', type=str, metavar='FILENAME', required=False,
                        help="Name of compressed file")
    parser.add_argument('--dry-run', dest='dryrun', action='store_true', help='Perform a dry run')
    parser.add_argument('-e', '--exclude', metavar='EXCLUDE',  nargs='+',
                        help="Name of subdirectories or files to exclude")
    # Load data and model configs
    cmd_args = parser.parse_args()
 
    main(dirname=cmd_args.dir, 
         compress_name=cmd_args.name, 
         exclude=cmd_args.exclude, 
         dryrun=cmd_args.dryrun)
    
#   python compress_exps.py --dir exps/v5/pngs-all/novice/ --name pngs-all-subset -e graph-adj* metrics* .*-scores* params* rq.*\.pdf ins outs .*-precision .*gamma= directed-adj directed-graph-adj graph-adj ^estimated-graph.pdf

# python compress_exps.py --dir exps/fc --name fc-subset -e graph-adj* metrics* gamma* graph-precision* params* ebic fb-score ground-truth rq.*\.pdf refined-precision refined-graph mpl