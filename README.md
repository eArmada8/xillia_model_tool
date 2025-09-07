# Tales of Xillia (PS3) mesh export
A script to get the mesh data out of the files from Tales of Xillia and Xillia 2 (PS3).  The output is in .glb files, although there is an option for .fmt/.ib/.vb/.vgmap that are compatible with DarkStarSword Blender import plugin for 3DMigoto.

## Credits:
I am as always very thankful for the dedicated reverse engineers at the Tales of ABCDE discord and the Kiseki modding discord, for their brilliant work, and for sharing that work so freely.

## Requirements:
1. Python 3.10 and newer is required for use of these scripts.  It is free from the Microsoft Store, for Windows users.  For Linux users, please consult your distro.
2. The numpy module for python is needed.  Install by typing "python3 -m pip install numpy" in the command line / shell.  (The struct, json, glob, copy, subprocess, os, sys, and argparse modules are also required, but these are all already included in most basic python installations.)
3. The output can be imported into Blender as .glb, or as raw buffers using DarkStarSword's amazing plugin: https://github.com/DarkStarSword/3d-fixes/blob/master/blender_3dmigoto.py (tested on commit [5fd206c](https://raw.githubusercontent.com/DarkStarSword/3d-fixes/5fd206c52fb8c510727d1d3e4caeb95dac807fb2/blender_3dmigoto.py))
4. xillia_export_model.py is dependent on lib_fmtibvb.py, which must be in the same folder.
5. [TalesOfTool](https://github.com/DaZombieKiller/TalesOfTools) is required for extracting and renaming files

## Usage:
### xillia_export_model.py
Double click the python script and it will search for all model files (a matching set of .TOSHPB, .TOSHPP, .TOSHPS files) to export them as .glb.  It will search all the available skeleton files (.TOHRCB files) for a skeleton.  This script expects all the necessary files to be available in the same folder; I recommending copying the files and skeleton to their own folder.  Textures should be placed in a `textures` folder.

See the `xillia_tltool_tex2dds.py` section below for instructions on extracting Tales of Xillia assets using tltool.exe.

**Command line arguments:**
`xillia_export_model.py [-h] [-t] [-d] [-o] shpb_file`

`-t, --textformat`
Output .gltf/.bin format instead of .glb format.

`-d, --dumprawbuffers`
Dump .fmt/.ib/.vb/.vgmap files in a folder with the same name as the .mdl file.  Use DarkStarSword's plugin to view.

`-h, --help`
Shows help message.

`-o, --overwrite`
Overwrite existing files without prompting.

### xillia_tltool_tex2dds.py
This script will use tltool.exe (included in [Tales Of Tools](https://github.com/DaZombieKiller/TalesOfTools) by DaZombieKiller) to convert all the textures for Tales of Xillia into .dds format.  First, to extract the gamedata, place this file in the folder with FILEHEADER.TOFHDB and TLFILE.TLDAT and extract with `tltool.exe unpack FILEHEADER.TOFHDB TLFILE.TLDAT extracted_files --bit32 --big-endian` (and add `--dictionary x1_filelist.log` to the end of that command if using the filelist by Meebo available at the Tales of ABCDE discord).  Then run `python xillia_tltool_tex2dds.py` and it will convert all the .TOTEXB/.TOTEXP files into .dds files and place them all in a `dds` folder in the extracted files folder.
