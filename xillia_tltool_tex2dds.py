# Tool to batch export textures from Tales of Xillia (PS3) using tltool.exe
# included in (Tales of Tools) by DaZombieKiller.
#
# Usage:  First extract all the files from the TLFILE.TLDAT archive using
# tltool.exe, then place this script in the folder with tltool.exe and run.
#
# This script is dependent on tltool.exe by DaZombieKiller, download here:
# https://github.com/DaZombieKiller/TalesOfTools/releases
#
# GitHub eArmada8/xillia_model_tool

try:
    import glob, subprocess, os, sys
except ModuleNotFoundError as e:
    print("Python module missing! {}".format(e.msg))
    input("Press Enter to abort.")
    raise

if __name__ == "__main__":
    # Set current directory
    if getattr(sys, 'frozen', False):
        os.chdir(os.path.dirname(sys.executable))
    else:
        os.chdir(os.path.abspath(os.path.dirname(__file__)))

    platform = 'ps3'
    print("Searching for texture files...")
    totexb = glob.glob('**/*.TOTEXB', recursive=True)
    totexp = glob.glob('**/*.TOTEXP', recursive=True)
    basenames = [os.path.basename(x[:-7]) for x in totexb]
    targets = {}
    for name in basenames:
        if any([name in x for x in totexb]) and any([name in x for x in totexp]):
            entry = {}
            entry['b'] = [x for x in totexb if name in x][0]
            entry['p'] = [x for x in totexp if name in x][0]
            entry['d'] = entry['b'].replace('TOTEXB', 'dds')
            targets[name] = entry
    print("Running tltool.exe...")
    for name in targets:
        if not os.path.exists(os.path.dirname(targets[name]['d'])):
            os.makedirs(os.path.dirname(targets[name]['d']))
        subprocess.run(['tltool.exe', 'tex2dds', targets[name]['b'], targets[name]['p'],
            targets[name]['d'], '--platform', platform])