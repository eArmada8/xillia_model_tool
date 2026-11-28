# Tool to export model data from the model format used by Tales of Xillia (PS3).
#
# Usage:  Run by itself without commandline arguments and it will search for model files
# (a matching set of .TOSHPB, .TOSHPP, .TOSHPS files) and export a .gib file.  It will do
# its best to find the matching skeleton file from the .TOHRCB files in the folder.
#
# For command line options, run:
# /path/to/python3 xillia_export_model.py --help
#
# Requires lib_fmtibvb.py, put in the same directory
#
# GitHub eArmada8/xillia_model_tool

try:
    import struct, json, math, numpy, copy, io, glob, os, sys
    from lib_fmtibvb import *
except ModuleNotFoundError as e:
    print("Python module missing! {}".format(e.msg))
    input("Press Enter to abort.")
    raise

# Configuration variables:

# True to enable combining models (requires a compatible skeleton, no commandline arguments)
combine_models_into_single_gltf = True

# Convert the model from left-handed to right-handed (from Tales space to glTF space)
flip_model_handedness_for_gltf = True

# Global variable, do not edit
e = '<'

def set_endianness (endianness):
    global e
    if endianness in ['<', '>']:
        e = endianness
    return

def read_offset (f):
    start_offset = f.tell()
    diff_offset, = struct.unpack("{}I".format(e), f.read(4))
    return(start_offset + diff_offset)

def read_string (f, start_offset):
    current_loc = f.tell()
    f.seek(start_offset)
    null_term_string = f.read(1)
    while null_term_string[-1] != 0:
        null_term_string += f.read(1)
    f.seek(current_loc)
    return(null_term_string[:-1].decode())

def trianglestrip_to_list(ib_list):
    triangles = []
    split_lists = [[]]
    # Split ib_list by primitive restart command, some models have this
    for i in range(len(ib_list)):
        if not ib_list[i] == 0xFFFF:
            split_lists[-1].append(ib_list[i])
        else:
            split_lists.append([])
    for i in range(len(split_lists)):
        for j in range(len(split_lists[i])-2):
            if j % 2 == 0:
                triangles.append([split_lists[i][j], split_lists[i][j+1], split_lists[i][j+2]])
            else:
                triangles.append([split_lists[i][j], split_lists[i][j+2], split_lists[i][j+1]]) #DirectX implementation
                #triangles.append([split_lists[i][j+1], split_lists[i][j], split_lists[i][j+2]]) #OpenGL implementation
    # Remove degenerate triangles
    triangles = [x for x in triangles if len(set(x)) == 3]
    return(triangles)

def read_skel_file (hrcb_file):
    skel_struct = []
    if len(hrcb_file) > 0 and os.path.exists(hrcb_file):
        with open(hrcb_file,'rb') as f:
            magic = f.read(4)
            if magic in [b'CRHM', b'MHRC']:
                set_endianness({b'CRHM': '<', b'MHRC': '>'}[magic])
                eof, unk = struct.unpack("{}2I".format(e), f.read(8))
                toc = [read_offset(f) for _ in range(6)]
                f.seek(toc[0])
                unk0 = struct.unpack("{}16H".format(e), f.read(32))
                unk1 = struct.unpack("{}f2I2fI2f".format(e), f.read(32))
                bone_count_0 = unk0[0] # Not sure why this number is larger
                bone_count_1 = unk0[1] # These seem to be the real bones with matrices and parentage
                inv_mtx = [struct.unpack("{}16f".format(e), f.read(64)) for _ in range(bone_count_1)] # Stored correctly
                abs_mtx = [struct.unpack("{}16f".format(e), f.read(64)) for _ in range(bone_count_1)] # Stored transposed
                abs_mtx_flip = [numpy.array(abs_mtx[i]).reshape(4,4).flatten('F').tolist() for i in range(len(abs_mtx))] # Column major
                dat2 = [struct.unpack("{}I2HI2HI2HI2H".format(e), f.read(32)) for _ in range(bone_count_1)] # all the numbers are the same except ID number
                dat3 = [struct.unpack("{}I2H".format(e), f.read(8)) for _ in range(unk0[2])] # are these root nodes?
                dat4 = list(struct.unpack("{}{}I".format(e, bone_count_0), f.read(bone_count_0 * 4))) # list of bones, equal to first val of dat2
                dat5 = [struct.unpack("{}2h".format(e), f.read(4)) for _ in range(bone_count_0)] # the second value is -1 unless there is an entry in dat3 - again, root node info?
                dat6 = [struct.unpack("{}2h3f".format(e), f.read(16)) for _ in range(bone_count_1)] # first value is probably parent, second value is 0 if in dat3, 1 if not
                # We need to skip the long list of floats, because I have no idea how to determine the length.  We will search for the first reasonably valid pointer
                val, = struct.unpack("{}I".format(e), f.read(4))
                while (val > 0xFFFF or val < 0x1):
                    val, = struct.unpack("{}I".format(e), f.read(4))
                f.seek(-4,1)
                names = []
                for _ in range(bone_count_1):
                    #3x u32 - start offset, end offset, blank
                    names.append(read_string(f, read_offset(f)))
                    f.seek(8,1) # skip end offset and blank
                skel_struct = [{'id': dat2[i][0], 'name': names[i], 'abs_matrix': abs_mtx_flip[i],\
                'inv_matrix': inv_mtx[i], 'parent': dat6[i][0]} for i in range(bone_count_1)]
                for i in range(len(skel_struct)):
                    if skel_struct[i]['parent'] in range(len(skel_struct)):
                        abs_mtx = [skel_struct[i]['abs_matrix'][0:4], skel_struct[i]['abs_matrix'][4:8],\
                            skel_struct[i]['abs_matrix'][8:12], skel_struct[i]['abs_matrix'][12:16]]
                        parent_inv_mtx = [skel_struct[skel_struct[i]['parent']]['inv_matrix'][0:4],\
                            skel_struct[skel_struct[i]['parent']]['inv_matrix'][4:8],\
                            skel_struct[skel_struct[i]['parent']]['inv_matrix'][8:12],\
                            skel_struct[skel_struct[i]['parent']]['inv_matrix'][12:16]]
                        skel_struct[i]['matrix'] = numpy.dot(abs_mtx, parent_inv_mtx).flatten('C').tolist()
                    else:
                        skel_struct[i]['matrix'] = skel_struct[i]['abs_matrix']
                    skel_struct[i]['children'] = [j for j in range(len(skel_struct)) if skel_struct[j]['parent'] == i]
    return(skel_struct)

def find_skeleton (bone_palette_ids):
    print("Searching all hrcb files for primary skeleton in current folder.")
    hrcb_files = glob.glob('*.TOHRCB')
    if len(hrcb_files) > 10:
        print("This may take a long time...")
    palettes = {}
    failed_to_read = []
    for i in range(len(hrcb_files)):
        try:
            skel_struct = read_skel_file(hrcb_files[i])
            palettes[hrcb_files[i]] = [x['id'] for x in skel_struct]
        except:
            print("Attempt to read skeleton file {} failed, skipping...".format(hrcb_files[i]))
            failed_to_read.append(i)
            pass
    for i in range(len(failed_to_read), 0, -1):
        hrcb_files.pop(failed_to_read[i-1])
    matches = [x for x in hrcb_files if all([y in palettes[x] for y in bone_palette_ids])]
    match = ''
    if len(matches) > 1:
        print("Multiple matches found, please choose one.")
        for i in range(len(matches)):
            print("{0}. {1}".format(i+1, matches[i]))
            if (i+1) % 25 == 0:
                input("More results, press Enter to continue...")
        while match == '':
            raw_input = input("Use which skeleton? ")
            if raw_input.isnumeric() and int(raw_input)-1 in range(len(matches)):
                match = matches[int(raw_input)-1]
            else:
                print("Invalid entry!")
    elif len(matches) == 1:
        match = matches[0]
    if match == '':
        print("No matches found!")
    else:
        print("Using {} as primary skeleton.".format(match))
    return match

def make_fmt(num_uvs, semantics_present = {'NORMAL': True, 'TANGENT': False, 'BINORMAL': False,
        'COLOR': False, 'BLENDWEIGHT': True, 'BLENDINDICES': True}):
    fmt = {'stride': '0', 'topology': 'trianglelist', 'format':\
        "DXGI_FORMAT_R16_UINT", 'elements': []}
    element_id, stride = 0, 0
    semantic_index = {'TEXCOORD': 0} # Counters for multiple indicies
    elements = []
    for i in range(7 + num_uvs):
            # I think order matters in this dict, so we will define the entire structure with default values
            element = {'id': '{0}'.format(element_id), 'SemanticName': '', 'SemanticIndex': '0',\
                'Format': '', 'InputSlot': '0', 'AlignedByteOffset': str(stride),\
                'InputSlotClass': 'per-vertex', 'InstanceDataStepRate': '0'}
            semantic_present = False
            if i == 0:
                element['SemanticName'] = 'POSITION'
                element['Format'] = 'R32G32B32_FLOAT'
                element_id += 1
                stride += 12
                semantic_present = True
            elif i == 1 and semantics_present['NORMAL']:
                element['SemanticName'] = 'NORMAL'
                element['Format'] = 'R32G32B32_FLOAT'
                element_id += 1
                stride += 12
                semantic_present = True
            elif i == 2 and semantics_present['TANGENT']:
                element['SemanticName'] = 'TANGENT'
                element['Format'] = 'R32G32B32_FLOAT'
                element_id += 1
                stride += 12
                semantic_present = True
            elif i == 3 and semantics_present['BINORMAL']:
                element['SemanticName'] = 'BINORMAL'
                element['Format'] = 'R32G32B32_FLOAT'
                element_id += 1
                stride += 12
                semantic_present = True
            elif i == 4 and semantics_present['COLOR']:
                element['SemanticName'] = 'COLOR'
                element['Format'] = 'R8G8B8A8_UNORM'
                element_id += 1
                stride += 4
                semantic_present = True
            elif i == 7+num_uvs-2 and semantics_present['BLENDWEIGHT']:
                element['SemanticName'] = 'BLENDWEIGHT'
                element['Format'] = 'R32G32B32A32_FLOAT'
                element_id += 1
                stride += 16
                semantic_present = True
            elif i == 7+num_uvs-1 and semantics_present['BLENDINDICES']:
                element['SemanticName'] = 'BLENDINDICES'
                element['Format'] = 'R8G8B8A8_UINT'
                element_id += 1
                stride += 4
                semantic_present = True
            elif i > 4 and i < (5 + num_uvs):
                element['SemanticName'] = 'TEXCOORD'
                element['SemanticIndex'] = str(semantic_index['TEXCOORD'])
                element['Format'] = 'R32G32_FLOAT'
                semantic_index['TEXCOORD'] += 1
                element_id += 1
                stride += 8
                semantic_present = True
            if semantic_present == True:
                elements.append(element)
    fmt['stride'] = str(stride)
    fmt['elements'] = elements
    return(fmt)

def read_mesh (mesh_info, vt_f, idx_f, morph_target_data = []):
    def read_floats (f, num):
        return(list(struct.unpack("{}{}f".format(e, num), f.read(num * 4))))
    def read_bytes (f, num):
        return(list(struct.unpack("{}{}B".format(e, num), f.read(num))))
    def read_interleaved_floats (f, num, stride, total):
        vecs = []
        padding = stride - (num * 4)
        for i in range(total):
            vecs.append(read_floats(f, num))
            if i < (total - 1):
                f.seek(padding, 1)
        return(vecs)
    def read_interleaved_bytes (f, num, stride, total):
        vecs = []
        padding = stride - (num)
        for i in range(total):
            vecs.append(read_bytes(f, num))
            if i < (total - 1):
                f.seek(padding, 1)
        return(vecs)
    def read_interleaved_ufloats (f, num, stride, total): # Unsigned normalized byte floats
        vecs = []
        padding = stride - (num)
        float_max = ((2**8)-1)
        for i in range(total):
            vals = read_bytes(f, num)
            vecs.append([x / float_max for x in vals])
            if i < (total - 1):
                f.seek(padding, 1)
        return(vecs)
    def read_vert_vector_data (f, flags, stride, num_verts, nrm_offset = 0, tangent_offset = 0, binormal_offset = 0):
        positions, normals, tangents, binormals = [], [], [], []
        base_start = f.tell()
        positions = read_interleaved_floats(f, 3, stride, num_verts)
        if not flags & 0x2000:
            f.seek(base_start + nrm_offset)
            normals = read_interleaved_floats(f, 3, stride, num_verts)
            if flags & 0x80:
                f.seek(base_start + tangent_offset)
                tangents = read_interleaved_floats(f, 3, stride, num_verts)
                f.seek(base_start + binormal_offset)
                binormals = read_interleaved_floats(f, 3, stride, num_verts)
        return positions, normals, tangents, binormals
    def fix_weights (weights):
        for _ in range(3):
            weights = [x+[round(1-sum(x),6)] if len(x) < 4 else x for x in weights]
        return(weights)
    vt_start_offset = mesh_info['vert_offset']
    num_verts = mesh_info['num_verts']
    idx_start_offset = mesh_info['idx_offset']
    num_idx = mesh_info['num_idx']
    offset_gpu_verts = mesh_info['uv_offset']
    flags = mesh_info['flags']
    # Calculate strides from flags
    semantics_present = {'NORMAL': False, 'TANGENT': False, 'BINORMAL': False,
            'COLOR': True, 'BLENDWEIGHT': False, 'BLENDINDICES': False} # Only color is always present
    cpu_stride_no_wt, gpu_stride = 0, 0 # final stride = cpu_stride_no_wt + weights, which are variable
    pos_offset, nrm_offset, tangent_offset, binormal_offset, blend_idx_offset, blend_wt_offset = 0, 0, 0, 0, 0, 0
    if flags & 0x30: # CPU skinning, vertex vector data in TOMDLB_D
        cpu_stride_no_wt += 12
        if not flags & 0x2000: # Unclear if this flag is actually used
            semantics_present['NORMAL'] = True
            nrm_offset = cpu_stride_no_wt
            cpu_stride_no_wt += 12
            if flags & 0x80:
                semantics_present['TANGENT'] = True
                semantics_present['BINORMAL'] = True
                tangent_offset = cpu_stride_no_wt
                cpu_stride_no_wt += 12
                binormal_offset = cpu_stride_no_wt
                cpu_stride_no_wt += 12
        elif flags & 0x20:
            nrm_offset = cpu_stride_no_wt
            cpu_stride_no_wt += 12 # If flags & 0x20, the normal slot is always there, even if unused
        if not (flags & 0x20):
            blend_idx_offset = cpu_stride_no_wt
            cpu_stride_no_wt += 4 # Blend indices
            blend_wt_offset = cpu_stride_no_wt
        if flags & 0x10:
            semantics_present['BLENDWEIGHT'] = True
            semantics_present['BLENDINDICES'] = True
    else: # No skinning, vertex vector data in TOMDLP_P (sent directly to GPU)
        gpu_stride += 12
        if not flags & 0x2000:
            semantics_present['NORMAL'] = True
            nrm_offset = gpu_stride
            gpu_stride += 12
            if flags & 0x80:
                semantics_present['TANGENT'] = True
                semantics_present['BINORMAL'] = True
                tangent_offset = gpu_stride
                gpu_stride += 12
                binormal_offset = gpu_stride
                gpu_stride += 12
    color_offset = gpu_stride
    gpu_stride += 4 # Vertex color
    uv_offsets = []
    num_uv_maps = flags & 0xF
    for _ in range(num_uv_maps):
        uv_offsets.append(gpu_stride)
        gpu_stride += 8 # UVs
    # Read vertex data
    positions, normals, tangents, binormals, blend_idx, weights, morphs = [], [], [], [], [], [], []
    vt_f.seek(vt_start_offset)
    if flags & 0x30:
        first_morph, total_morph, count, unk2 = struct.unpack("{}2H2I".format(e), vt_f.read(12))
        if flags & 0x20: # Morph target animation
            bone_slot, unk3, vert_count_1, vert_count_2 = list(struct.unpack("{}4H".format(e), vt_f.read(8)))
            base_offset = vt_f.tell()
            positions, normals, tangents, binormals = read_vert_vector_data (vt_f, flags, cpu_stride_no_wt, num_verts,
                nrm_offset, tangent_offset, binormal_offset)
            if flags & 0x10: # Skinned morph targets
                blend_idx = [[bone_slot, 0, 0, 0] for _ in range(num_verts)]
                weights = [[1.0, 0.0, 0.0, 0.0] for _ in range(num_verts)]
            for i in range(total_morph):
                dense_morph_block = morph_target_data[first_morph + i]['morph_run_binary']
                dense_morph_block.extend(b'\x00' * vert_count_2 * cpu_stride_no_wt)
                with io.BytesIO(dense_morph_block) as dense_morph_block_f:
                    morph_p, morph_n, morph_t, morph_b = read_vert_vector_data (dense_morph_block_f, flags,
                        cpu_stride_no_wt,num_verts, nrm_offset, tangent_offset, binormal_offset)
                morphs.append({'POSITION': morph_p, 'NORMAL': morph_n, 'TANGENT': morph_t, 'BINORMAL': morph_b,
                    'name': morph_target_data[first_morph + i]['name']})
        else: # Skeletal animation
            weights = []
            num_vertices_array = []
            for i in range(count):
                num_vertices_array.append(list(struct.unpack("{}4H".format(e), vt_f.read(8))))
            for i in range(len(num_vertices_array)):
                for j in range(len(num_vertices_array[i])):
                    if num_vertices_array[i][j] > 0:
                        base_start = vt_f.tell()
                        stride = cpu_stride_no_wt + (j * 4)
                        new_p, new_n, new_t, new_b = read_vert_vector_data (vt_f, flags, stride,
                            num_vertices_array[i][j], nrm_offset, tangent_offset, binormal_offset)
                        positions.extend(new_p)
                        normals.extend(new_n)
                        tangents.extend(new_t)
                        binormals.extend(new_b)
                        vt_f.seek(base_start + blend_idx_offset)
                        blend_idx.extend(read_interleaved_bytes(vt_f, 4, stride, num_vertices_array[i][j]))
                        if j > 0:
                            vt_f.seek(base_start + blend_wt_offset)
                            weights.extend(read_interleaved_floats(vt_f, j, stride, num_vertices_array[i][j]))
                        else:
                            weights.extend([[1.0] for _ in range(num_vertices_array[i][j])])
            weights = fix_weights(weights)
    else:
        idx_f.seek(offset_gpu_verts)
        positions, normals, tangents, binormals = read_vert_vector_data (idx_f, flags, gpu_stride, num_verts,
            nrm_offset, tangent_offset, binormal_offset)
    idx_f.seek(offset_gpu_verts + color_offset)
    colors = read_interleaved_ufloats(idx_f, 4, gpu_stride, num_verts)
    uv_maps = []
    for i in range(len(uv_offsets)):
        idx_f.seek(offset_gpu_verts + uv_offsets[i])
        uv_maps.append(read_interleaved_floats (idx_f, 2, gpu_stride, num_verts))
    # Read index data
    idx_f.seek(idx_start_offset)
    idx_buffer = list(struct.unpack("{}{}H".format(e, num_idx), idx_f.read(num_idx * 2)))
    if not (flags & 0xF00 == 0x100):
        ib = trianglestrip_to_list(idx_buffer)
    else:
        ib = idx_buffer
    # Build raw buffers for export
    fmt = make_fmt(len(uv_maps), semantics_present)
    vb = [{'Buffer': positions}]
    if not flags & 0x2000:
        vb.append({'Buffer': normals})
    if flags & 0x80:
        vb.append({'Buffer': tangents})
        vb.append({'Buffer': binormals})
    vb.append({'Buffer': colors})
    for uv_map in uv_maps:
        vb.append({'Buffer': uv_map})
    if flags & 0x10:
        vb.append({'Buffer': weights})
        vb.append({'Buffer': blend_idx})
    mesh = {'fmt': fmt, 'vb': vb, 'ib': ib, 'morphs': morphs, 'mesh_info': mesh_info}
    if flags & 0x80:
        expected_binormals = numpy.cross(normals, tangents)
        w = numpy.where(numpy.einsum('ij,ij->i', expected_binormals, binormals) >= 0.0, 1.0, -1.0)
        mesh['tangent4'] = [tangents[i] + [float(w[i])] for i in range(len(tangents))]
    return(mesh)

#Meshes, offset should be toc[0].  Requires shps filename for verts, and shpp filename for uv's and index buffer.
def read_section_1 (f, offset, shps_file, shpp_file, morph_target_data = []):
    f.seek(offset)
    mesh_count, palette_count = struct.unpack("{}2H".format(e), f.read(4))
    set_0 = [struct.unpack("{}I3f".format(e), f.read(16)) for _ in range(mesh_count)]
    set_1 = []
    names = []
    for _ in range(mesh_count):
        set_1.append(struct.unpack("{}I4H".format(e), f.read(12)))
        names.append(read_string(f, read_offset (f)))
    mesh_blocks_info = []
    for i in range(mesh_count):
        mesh_info = {'name': names[i]}
        mesh_info['num_verts'], mesh_info['num_idx'], mesh_info['uv_offset'], mesh_info['idx_offset'], mesh_info['vert_offset'] = \
            struct.unpack("{}2H3I".format(e), f.read(16))
        mesh_info['mesh'] = set_1[i][0]
        mesh_info['submesh'] = set_1[i][1]
        mesh_info['flags'] = set_1[i][2]
        mesh_info['material'] = set_1[i][3]
        mesh_info['material_flags'] = set_1[i][4]
        mesh_info['mesh_sort'] = set_0[i]
        mesh_blocks_info.append(mesh_info)
    bone_palette_ids = struct.unpack("{}{}I".format(e, palette_count), f.read(4 * palette_count))
    meshes = []
    with open (shps_file, 'rb') as vt_f:
        with open (shpp_file, 'rb') as idx_f:
            for i in range(mesh_count):
                meshes.append(read_mesh(mesh_blocks_info[i], vt_f, idx_f, morph_target_data))
    return(meshes, bone_palette_ids, mesh_blocks_info)

def read_morph_run (offset, vt_f):
    vt_f.seek(offset)
    unk = struct.unpack("{}3I".format(e), vt_f.read(12))
    size, = struct.unpack("{}H2x".format(e), vt_f.read(4))
    dense_morph_block = bytearray()
    while True:
        skip, count = struct.unpack("{}2I".format(e), vt_f.read(8))
        dense_morph_block.extend(b'\x00' * 12 * skip)
        if count > 0:
            dense_morph_block.extend(vt_f.read(12 * count))
        else:
            break
    return (unk, dense_morph_block)

#Morph Targets, offset should be toc[1]. 
def read_section_2 (f, offset, shps_file):
    def decode_target_flag (flag):
        return({'type': flag & 0xFFFFFFFF, 'target': flag >> 32 & 0xFFFFFF, 'category': flag >> 56})
    f.seek(offset)
    counts = struct.unpack("{}2H".format(e), f.read(4))
    f.seek(4,1) # Padding?
    targets = struct.unpack("{}{}Q".format(e, counts[1]), f.read(8 * counts[1]))
    targets_decoded = [decode_target_flag(x) for x in targets]
    morph_target_data = []
    for i in range(counts[0]):
        name = read_string(f, read_offset(f))
        buffer_offset, id_ = struct.unpack("{}2I".format(e), f.read(8))
        morph_target_data.append({'name': name, 'offset': buffer_offset, 'id': id_, 'target': targets_decoded[i]})
    unk_floats = struct.unpack("{}{}f".format(e, counts[0]), f.read(4 * counts[0]))
    for i in range(counts[0]):
        morph_target_data[i]['unk_float'] = unk_floats[i]
    with open (shps_file, 'rb') as vt_f:
        for i in range(counts[0]):
            unk, dense_morph_block = read_morph_run(morph_target_data[i]['offset'], vt_f)
            morph_target_data[i]['unk_morph_data'] = unk
            morph_target_data[i]['morph_run_binary'] = dense_morph_block
    return morph_target_data

def make_morphed_meshes (mesh):
    morphed_meshes = []
    for i in range(len(mesh['morphs'])):
        morphed_vb = []
        for j in range(len(mesh['vb'])):
            semantic = mesh['fmt']['elements'][j]['SemanticName']
            if semantic in mesh['morphs'][i]:
                buffer = []
                for k in range(len(mesh['vb'][j]['Buffer'])):
                    buffer.append([mesh['vb'][j]['Buffer'][k][l] + mesh['morphs'][i][semantic][k][l]
                        for l in range(len(mesh['vb'][j]['Buffer'][k]))])
                morphed_vb.append({'Buffer': buffer})
            else:
                morphed_vb.append(mesh['vb'][j])
        morphed_meshes.append({'fmt': mesh['fmt'], 'vb': morphed_vb, 'ib': mesh['ib'], 'name': mesh['morphs'][i]['name']})
    return(morphed_meshes)

#Materials, offset should be toc[2]. 
def read_section_3 (f, offset):
    f.seek(offset)
    counts = struct.unpack("{}8H".format(e), f.read(16))
    mat_count = counts[0]
    tex_pointer_count = counts[1]
    unk_count_1 = counts[2]
    unk_count_2 = counts[3]
    unk_count_3a = unk_count_1 + unk_count_2
    unk_count_3 = unk_count_3a + 1 if unk_count_3a % 2 else unk_count_3a # Round up to nearest even number
    tex_count = counts[4]
    unk_header_count = counts[5]
    for _ in range(unk_header_count):
        f.seek(8,1)
    set_0 = []
    for i in range(mat_count):
        val1 = list(struct.unpack("{}I8h".format(e), f.read(20)))
        offset1 = read_offset(f)
        if i == 0:
            end_offset = offset1
        offset2 = read_offset(f)
        current = f.tell()
        f.seek(offset1)
        val2a = list(struct.unpack("{}10I".format(e), f.read(40)))
        assert f.tell() == offset2
        val2b = list(struct.unpack("{}5I4fI".format(e), f.read(40)))
        set_0.append({'base': val1, 'val2': [val2a,val2b]})
        f.seek(current)
    set_1 = [struct.unpack("{}4h".format(e), f.read(8)) for _ in range(tex_pointer_count)]
    set_2 = []
    for _ in range(mat_count):
        offset1 = read_offset(f)
        offset2 = read_offset(f)
        offset3 = read_offset(f)
        offset4 = read_offset(f)
        current = f.tell()
        f.seek(offset1)
        vals = list(struct.unpack("{}4I".format(e), f.read(16)))
        name = read_string(f, offset3)
        set_2.append({'name': name, 'vals': vals})
        f.seek(current)
    set_3 = [struct.unpack("{}4f".format(e), f.read(16)) for _ in range(unk_count_1)]
    set_4 = [struct.unpack("{}6f".format(e), f.read(24)) for _ in range(unk_count_2)]
    set_5 = list(struct.unpack("{}{}h".format(e, unk_count_3), f.read(unk_count_3 *2)))
    textures = []
    for _ in range(tex_count):
        textures.append(read_string(f, read_offset(f)))
    material_struct = []
    if len(set_0) == len(set_2):
        for i in range(len(set_0)):
            material = {'name': set_2[i]['name']}
            material['textures'] = [textures[set_1[set_0[i]['base'][5]+j][1]] for j in range(set_0[i]['base'][3])]
            material['alpha'] = 1 if set_0[i]['base'][2] > 0 else 0
            material['internal_id'] = set_0[i]['base'][0]
            material['unk_parameters'] = {'set_0_base': [set_0[i]['base'][1], set_0[i]['base'][4]] + set_0[i]['base'][6:],
                'set_0_values': set_0[i]['val2'], 'set_2_values': set_2[i]['vals']}
            material_struct.append(material)
    return (material_struct)

def convert_format_for_gltf(dxgi_format):
    dxgi_format = dxgi_format.split('DXGI_FORMAT_')[-1]
    dxgi_format_split = dxgi_format.split('_')
    if len(dxgi_format_split) == 2:
        numtype = dxgi_format_split[1]
        vec_format = re.findall("[0-9]+",dxgi_format_split[0])
        vec_bits = int(vec_format[0])
        vec_elements = len(vec_format)
        if numtype in ['FLOAT', 'UNORM', 'SNORM']:
            componentType = 5126
            componentStride = len(re.findall('[0-9]+', dxgi_format)) * 4
            dxgi_format = "".join(['R32','G32','B32','A32','D32'][0:componentStride//4]) + "_FLOAT"
        elif numtype == 'UINT':
            if vec_bits == 32:
                componentType = 5125
                componentStride = len(re.findall('[0-9]+', dxgi_format)) * 4
            elif vec_bits == 16:
                componentType = 5123
                componentStride = len(re.findall('[0-9]+', dxgi_format)) * 2
            elif vec_bits == 8:
                componentType = 5121
                componentStride = len(re.findall('[0-9]+', dxgi_format))
        accessor_types = ["SCALAR", "VEC2", "VEC3", "VEC4"]
        accessor_type = accessor_types[len(re.findall('[0-9]+', dxgi_format))-1]
        return({'format': dxgi_format, 'componentType': componentType,\
            'componentStride': componentStride, 'accessor_type': accessor_type})
    else:
        return(False)

def convert_fmt_for_gltf(fmt):
    new_fmt = copy.deepcopy(fmt)
    stride = 0
    new_semantics = {'BLENDWEIGHT': 'WEIGHTS', 'BLENDINDICES': 'JOINTS'}
    need_index = ['WEIGHTS', 'JOINTS', 'COLOR', 'TEXCOORD']
    for i in range(len(fmt['elements'])):
        if new_fmt['elements'][i]['SemanticName'] in new_semantics.keys():
            new_fmt['elements'][i]['SemanticName'] = new_semantics[new_fmt['elements'][i]['SemanticName']]
        new_info = convert_format_for_gltf(fmt['elements'][i]['Format'])
        new_fmt['elements'][i]['Format'] = new_info['format']
        if new_fmt['elements'][i]['SemanticName'] in need_index:
            new_fmt['elements'][i]['SemanticName'] = new_fmt['elements'][i]['SemanticName'] + '_' +\
                new_fmt['elements'][i]['SemanticIndex']
        new_fmt['elements'][i]['AlignedByteOffset'] = stride
        new_fmt['elements'][i]['componentType'] = new_info['componentType']
        new_fmt['elements'][i]['componentStride'] = new_info['componentStride']
        new_fmt['elements'][i]['accessor_type'] = new_info['accessor_type']
        stride += new_info['componentStride']
    index_fmt = convert_format_for_gltf(fmt['format'])
    new_fmt['format'] = index_fmt['format']
    new_fmt['componentType'] = index_fmt['componentType']
    new_fmt['componentStride'] = index_fmt['componentStride']
    new_fmt['accessor_type'] = index_fmt['accessor_type']
    new_fmt['stride'] = stride
    return(new_fmt)

def flip_mesh_handedness_for_gltf (submesh):
    # Technically binormals aren't used but in case glTF ever changes
    to_flip = ['POSITION', 'NORMAL', 'TANGENT', 'BINORMAL']
    for i in range(len(submesh['vb'])):
        semantic = submesh['fmt']['elements'][i]['SemanticName']
        if semantic in to_flip:
            submesh['vb'][i]['Buffer'] = [[-x if (i == 2 or (i == 3 and semantic == 'TANGENT')) else x
                for i, x in enumerate(vec)] for vec in submesh['vb'][i]['Buffer']]
    # Reverse the unwinding order of the triangles
    if len(submesh['ib']) > 0 and isinstance(submesh['ib'][0], list):
        submesh['ib'] = [[x[2], x[1], x[0]] for x in submesh['ib']]
    else:
        submesh['ib'].reverse()
    # Flip morph targets
    for i in range(len(submesh['morphs'])):
        for semantic in [x for x in to_flip if x in submesh['morphs'][i]]:
            submesh['morphs'][i][semantic] = [[-x if i == 2 else x
                for i, x in enumerate(vec)] for vec in submesh['morphs'][i][semantic]]
    return(submesh)

def convert_matrix_to_trs (matrix):
    scale = [numpy.linalg.norm(matrix[0:3]), numpy.linalg.norm(matrix[4:7]),\
        numpy.linalg.norm(matrix[8:11])]
    r_mtx = numpy.array([(matrix[0:3]/scale[0]).tolist(),\
        (matrix[4:7]/scale[1]).tolist(),\
        (matrix[8:11]/scale[2]).tolist()]) # Row-major
    # Enforce orthogonality of rotation matrix, Premelani W and Bizard P "Direction Cosine Matrix IMU: Theory" Diy Drone: Usa 1 (2009).
    if (error := numpy.dot(r_mtx[0],r_mtx[1])) != 0.0:
        vectors = [r_mtx[0]-(error/2)*r_mtx[1], r_mtx[1]-(error/2)*r_mtx[0]]
        vectors.append(numpy.cross(vectors[0], vectors[1]))
        r_mtx = numpy.array([x/numpy.linalg.norm(x) for x in vectors])
    # Adapted from Day M. "Converting a Rotation Matrix to a Quaternion." Insomniac Games (13 Jan 2015)
    if (r_mtx[2][2] < 0):
        if (r_mtx[0][0] > r_mtx[1][1]):
            t = 1 + r_mtx[0][0] - r_mtx[1][1] - r_mtx[2][2]
            q = numpy.array([t, r_mtx[0][1]+r_mtx[1][0], r_mtx[2][0]+r_mtx[0][2], r_mtx[1][2]-r_mtx[2][1]])
        else:
            t = 1 - r_mtx[0][0] + r_mtx[1][1] - r_mtx[2][2]
            q = numpy.array([r_mtx[0][1]+r_mtx[1][0], t, r_mtx[1][2]+r_mtx[2][1], r_mtx[2][0]-r_mtx[0][2]])
    else:
        if (r_mtx[0][0] < -r_mtx[1][1]):
            t = 1 - r_mtx[0][0] - r_mtx[1][1] + r_mtx[2][2]
            q = numpy.array([r_mtx[2][0]+r_mtx[0][2], r_mtx[1][2]+r_mtx[2][1], t, r_mtx[0][1]-r_mtx[1][0]])
        else:
            t = 1 + r_mtx[0][0] + r_mtx[1][1] + r_mtx[2][2]
            q = numpy.array([r_mtx[1][2]-r_mtx[2][1], r_mtx[2][0]-r_mtx[0][2], r_mtx[0][1]-r_mtx[1][0], t])
    q *= 0.5 / math.sqrt(t) #xyzw
    return(matrix[12:15], q.tolist(), scale) #TRS

def flip_node_handedness_for_gltf (node):
    # Grab the original transform
    if 'matrix' in node:
        node['matrix'] = [-x if i in [2,6,8,9,11,14] else x for i, x in enumerate(node['matrix'])]
        return(node)
    else:
        t = node.get('translation', [0,0,0])
        r = node.get('rotation', [0,0,0,1])
        s = node.get('scale', [1,1,1])
    # Remove the original transform
    node.pop('matrix', None)
    node.pop('translation', None)
    node.pop('rotation', None)
    node.pop('scale', None)
    # Change handedness by flipping Z
    t = [t[0], t[1], -t[2]]
    r = [-r[0], -r[1], r[2], r[3]] #xyzw
    # Write back the new transform
    if not t == [0,0,0]:
        node['translation'] = t
    if not r == [0,0,0,1]:
        node['rotation'] = r #xyzw
    if not s == [1,1,1]:
        node['scale'] = s
    return(node)

def flip_ibm_handedness_for_gltf (ibm):
    return([-x if i in [2,6,8,9,11,14] else x for i, x in enumerate(ibm)])

def fix_strides(submesh):
    offset = 0
    for i in range(len(submesh['vb'])):
        submesh['fmt']['elements'][i]['AlignedByteOffset'] = str(offset)
        new_stride = get_stride_from_dxgi_format(submesh['fmt']['elements'][i]['Format'])
        offset += new_stride
    if 'vb0 stride' in submesh['fmt']:
        submesh['fmt']['vb0 stride'] = str(offset)
    else:
        submesh['fmt']['stride'] = str(offset)
    return(submesh)

def write_gltf(base_name, skel_struct, vgmaps, mesh_blocks_info, meshes, material_struct,\
        overwrite = False, write_binary_gltf = True):
    gltf_data = {}
    gltf_data['asset'] = { 'version': '2.0' }
    gltf_data['accessors'] = []
    gltf_data['bufferViews'] = []
    gltf_data['buffers'] = []
    gltf_data['meshes'] = []
    gltf_data['materials'] = []
    gltf_data['nodes'] = []
    gltf_data['samplers'] = []
    gltf_data['scenes'] = [{}]
    gltf_data['scene'] = 0
    gltf_data['skins'] = []
    gltf_data['textures'] = []
    giant_buffer = bytes()
    buffer_view = 0
    # Materials
    material_dict = [{'name': material_struct[i]['name'],
        'texture': material_struct[i]['textures'][0] if len(material_struct[i]['textures']) > 0 else '',
        'alpha': material_struct[i]['alpha'] if 'alpha' in material_struct[i] else 0}
        for i in range(len(material_struct))]
    texture_list = sorted(list(set([x['texture'] for x in material_dict if not x['texture'] == ''])))
    gltf_data['images'] = [{'uri':'textures/{}.dds'.format(x)} for x in texture_list]
    for mat in material_dict:
        material = { 'name': mat['name'] }
        material['pbrMetallicRoughness'] = { 'metallicFactor' : 0.0, 'roughnessFactor' : 1.0 }
        if not mat['texture'] == '':
            sampler = { 'wrapS': 10497, 'wrapT': 10497 } # I have no idea if this setting exists
            texture = { 'source': texture_list.index(mat['texture']), 'sampler': len(gltf_data['samplers']) }
            material['pbrMetallicRoughness']['baseColorTexture'] = { 'index' : len(gltf_data['textures']), }
            gltf_data['samplers'].append(sampler)
            gltf_data['textures'].append(texture)
        if mat['alpha'] & 1:
            material['alphaMode'] = 'MASK'
        gltf_data['materials'].append(material)
    material_list = [x['name'] for x in gltf_data['materials']]
    missing_textures = [x['uri'] for x in gltf_data['images'] if not os.path.exists(x['uri'])]
    if len(missing_textures) > 0:
        print("Warning:  The following textures were not found:")
        for texture in missing_textures:
            print("{}".format(texture))
    # Nodes
    node_id_to_g_id = {}
    for i in range(len(skel_struct)):
        g_node = {'children': skel_struct[i]['children'], 'name': skel_struct[i]['name'], 'matrix': skel_struct[i]['matrix']}
        if flip_model_handedness_for_gltf == True:
            g_node = flip_node_handedness_for_gltf(g_node)
        gltf_data['nodes'].append(g_node)
        node_id_to_g_id[skel_struct[i]['id']] = i
    for i in range(len(gltf_data['nodes'])):
        if len(gltf_data['nodes'][i]['children']) == 0 and i > 0:
            del(gltf_data['nodes'][i]['children'])
    if len(gltf_data['nodes']) == 0:
        gltf_data['nodes'].append({'children': [], 'name': 'root'})
    # World-space meshes will be attached to the first node as they don't have a home in the original model
    node_id_list = [x['id'] for x in skel_struct]
    for i in range(len(mesh_blocks_info)):
        mesh_blocks_info[i]['mesh_v'] = '{0}_{1:02d}'.format(mesh_blocks_info[i]['mesh'], mesh_blocks_info[i]['vgmap'])
    mesh_node_ids = {x['mesh'] if x['mesh'] in node_id_to_g_id else x['mesh_v']:x['name'] for x in mesh_blocks_info}
    if not 'children' in gltf_data['nodes'][0]:
        gltf_data['nodes'][0]['children'] = []
    for mesh_node_id in mesh_node_ids:
        if not mesh_node_id in node_id_list:
            g_node = {'name': mesh_node_ids[mesh_node_id]}
            gltf_data['nodes'][0]['children'].append(len(gltf_data['nodes']))
            gltf_data['nodes'].append(g_node)
    mesh_block_tree = {x:[i for i in range(len(mesh_blocks_info))
        if mesh_blocks_info[i]['mesh_v'] == x or mesh_blocks_info[i]['mesh'] == x] for x in mesh_node_ids}
    node_list = [x['name'] for x in gltf_data['nodes']]
    node_id_to_name = {x['id']:x['name'] for x in skel_struct}
    # Skin matrices
    skinning_possible = True
    try:
        ibms_struct = []
        inv_mtx_buffers = []
        for vgmap in vgmaps:
            vgmap_nodes = [node_list.index(x) for x in list(vgmap.keys())]
            ibms = [skel_struct[j]['inv_matrix'] for j in vgmap_nodes]
            if flip_model_handedness_for_gltf == True:
                ibms = [flip_ibm_handedness_for_gltf(x) for x in ibms]
            inv_mtx_buffer = b''.join([struct.pack("<16f", *x) for x in ibms])
            ibms_struct.append(ibms)
            inv_mtx_buffers.append(inv_mtx_buffer)
    except ValueError:
        skinning_possible = False
    # Meshes
    for mesh in mesh_block_tree: #Mesh
        primitives = []
        for j in range(len(mesh_block_tree[mesh])): #Submesh
            i = mesh_block_tree[mesh][j]
            # Vertex Buffer
            for k, element in enumerate(meshes[i]['fmt']['elements']): #Expand tangent to vec4
                if element['SemanticName'] == 'TANGENT' and 'tangent4' in meshes[i]:
                    meshes[i]['vb'][k]['Buffer'] = meshes[i]['tangent4']
                    element['Format'] = 'R32G32B32A32_FLOAT'
            meshes[i] = fix_strides(meshes[i])
            if flip_model_handedness_for_gltf == True:
                meshes[i] = flip_mesh_handedness_for_gltf (meshes[i])
            gltf_fmt = convert_fmt_for_gltf(meshes[i]['fmt'])
            vb_stream = io.BytesIO()
            write_vb_stream(meshes[i]['vb'], vb_stream, gltf_fmt, e='<', interleave = False)
            block_offset = len(giant_buffer)
            primitive = {"attributes":{}}
            for element in range(len(gltf_fmt['elements'])):
                if gltf_fmt['elements'][element]['SemanticName'] == 'BINORMAL':
                    gltf_fmt['elements'][element]['SemanticName'] = '_BINORMAL'
                primitive["attributes"][gltf_fmt['elements'][element]['SemanticName']]\
                    = len(gltf_data['accessors'])
                gltf_data['accessors'].append({"bufferView" : len(gltf_data['bufferViews']),\
                    "componentType": gltf_fmt['elements'][element]['componentType'],\
                    "count": len(meshes[i]['vb'][element]['Buffer']),\
                    "type": gltf_fmt['elements'][element]['accessor_type']})
                if gltf_fmt['elements'][element]['SemanticName'] == 'POSITION':
                    gltf_data['accessors'][-1]['max'] =\
                        [max([x[0] for x in meshes[i]['vb'][element]['Buffer']]),\
                         max([x[1] for x in meshes[i]['vb'][element]['Buffer']]),\
                         max([x[2] for x in meshes[i]['vb'][element]['Buffer']])]
                    gltf_data['accessors'][-1]['min'] =\
                        [min([x[0] for x in meshes[i]['vb'][element]['Buffer']]),\
                         min([x[1] for x in meshes[i]['vb'][element]['Buffer']]),\
                         min([x[2] for x in meshes[i]['vb'][element]['Buffer']])]
                gltf_data['bufferViews'].append({"buffer": 0,\
                    "byteOffset": block_offset,\
                    "byteLength": len(meshes[i]['vb'][element]['Buffer']) *\
                    gltf_fmt['elements'][element]['componentStride'],\
                    "target" : 34962})
                block_offset += len(meshes[i]['vb'][element]['Buffer']) *\
                    gltf_fmt['elements'][element]['componentStride']
            vb_stream.seek(0)
            giant_buffer += vb_stream.read()
            vb_stream.close()
            del(vb_stream)
            # Index Buffers
            ib_stream = io.BytesIO()
            write_ib_stream(meshes[i]['ib'], ib_stream, gltf_fmt, e='<')
            # IB is 16-bit so can be misaligned, unlike VB
            while (ib_stream.tell() % 4) > 0:
                ib_stream.write(b'\x00')
            primitive["indices"] = len(gltf_data['accessors'])
            gltf_data['accessors'].append({"bufferView" : len(gltf_data['bufferViews']),\
                "componentType": gltf_fmt['componentType'],\
                "count": len([index for triangle in meshes[i]['ib'] for index in triangle]),\
                "type": gltf_fmt['accessor_type']})
            gltf_data['bufferViews'].append({"buffer": 0,\
                "byteOffset": len(giant_buffer),\
                "byteLength": ib_stream.tell(),\
                "target" : 34963})
            ib_stream.seek(0)
            giant_buffer += ib_stream.read()
            ib_stream.close()
            del(ib_stream)
            primitive["mode"] = 4 #TRIANGLES
            primitive["material"] = mesh_blocks_info[i]['material']
            morph_weights = []
            if len(meshes[i]['morphs']) > 0:
                primitive["targets"] = []
                for morph in meshes[i]['morphs']:
                    target = {}
                    for element in morph:
                        if element == 'name':
                            continue
                        if len(morph[element]) > 0 and element != 'BINORMAL':
                            delta_block = numpy.array(morph[element], dtype='<f4').tobytes()
                            target[element] = len(gltf_data['accessors'])
                            gltf_data['accessors'].append({"bufferView" : len(gltf_data['bufferViews']),
                                "componentType": 5126,
                                "count": len(morph[element]),
                                "type": "VEC3"})
                            if element == 'POSITION':
                                gltf_data['accessors'][-1]['max'] =\
                                    [max([x[0] for x in morph[element]]),
                                     max([x[1] for x in morph[element]]),
                                     max([x[2] for x in morph[element]])]
                                gltf_data['accessors'][-1]['min'] =\
                                    [min([x[0] for x in morph[element]]),
                                     min([x[1] for x in morph[element]]),
                                     min([x[2] for x in morph[element]])]
                            gltf_data['bufferViews'].append({"buffer": 0,
                                "byteOffset": len(giant_buffer),
                                "byteLength": len(delta_block),
                                "target" : 34962})
                            giant_buffer += delta_block
                    primitive["targets"].append(target)
                    morph_weights.append(0.0)
            primitives.append(primitive)
        if len(primitives) > 0:
            if not meshes[i]['mesh_info']['mesh'] in node_id_to_g_id: # One of the new nodes
                node_id = node_list.index(mesh_node_ids[mesh])
            else: # One of the pre-assigned nodes (e.g. non-world-space mesh)
                node_id = node_id_to_g_id[meshes[i]['mesh_info']['mesh']]
            gltf_data['nodes'][node_id]['mesh'] = len(gltf_data['meshes'])
            gltf_data['meshes'].append({"primitives": primitives, "name": mesh_node_ids[mesh]})
            if len(morph_weights) > 0:
                gltf_data['meshes'][-1]['weights'] = morph_weights
            # Skinning
            if (len(vgmaps[mesh_blocks_info[i]["vgmap"]]) > 0 and skinning_possible == True
                    and all(['JOINTS_0' in p['attributes'] for p in primitives])
                    and (meshes[i]['mesh_info']['flags'] & 0x40)):
                gltf_data['nodes'][node_id]['skin'] = len(gltf_data['skins'])
                gltf_data['skins'].append({"inverseBindMatrices": len(gltf_data['accessors']),\
                    "joints": [node_list.index(x) for x in vgmaps[mesh_blocks_info[i]["vgmap"]]]})
                gltf_data['accessors'].append({"bufferView" : len(gltf_data['bufferViews']),\
                    "componentType": 5126,\
                    "count": len(ibms_struct[mesh_blocks_info[i]["vgmap"]]),\
                    "type": "MAT4"})
                gltf_data['bufferViews'].append({"buffer": 0,\
                    "byteOffset": len(giant_buffer),\
                    "byteLength": len(inv_mtx_buffers[mesh_blocks_info[i]["vgmap"]])})
                giant_buffer += inv_mtx_buffers[mesh_blocks_info[i]["vgmap"]]
    # Attach the root node (and any orphan nodes if the external skeleton is not available) to the scene
    child_nodes = set([x for y in gltf_data['nodes'] for x in y.get('children', [])])
    gltf_data['scenes'][0]['nodes'] = sorted([i for i in range(len(gltf_data['nodes'])) if i not in child_nodes])
    # Write GLB
    gltf_data['buffers'].append({"byteLength": len(giant_buffer)})
    if (os.path.exists(base_name + '.gltf') or os.path.exists(base_name + '.glb')) and (overwrite == False):
        if str(input(base_name + ".glb/.gltf exists! Overwrite? (y/N) ")).lower()[0:1] == 'y':
            overwrite = True
    if (overwrite == True) or not (os.path.exists(base_name + '.gltf') or os.path.exists(base_name + '.glb')):
        if write_binary_gltf == True:
            with open(base_name+'.glb', 'wb') as f:
                jsondata = json.dumps(gltf_data).encode('utf-8')
                jsondata += b' ' * (4 - len(jsondata) % 4)
                f.write(struct.pack('<III', 1179937895, 2, 12 + 8 + len(jsondata) + 8 + len(giant_buffer)))
                f.write(struct.pack('<II', len(jsondata), 1313821514))
                f.write(jsondata)
                f.write(struct.pack('<II', len(giant_buffer), 5130562))
                f.write(giant_buffer)
        else:
            gltf_data['buffers'][0]["uri"] = base_name+'.bin'
            with open(base_name+'.bin', 'wb') as f:
                f.write(giant_buffer)
            with open(base_name+'.gltf', 'wb') as f:
                f.write(json.dumps(gltf_data, indent=4).encode("utf-8"))

def process_shpb (shpb_file, overwrite = False, write_raw_buffers = False, write_binary_gltf = True):
    print("Processing {}...".format(shpb_file))
    with open(shpb_file, 'rb') as f:
        magic = f.read(4)
        if magic in [b'MPSM', b'MSPM']:
            set_endianness({b'MPSM': '<', b'MSPM': '>'}[magic])
            file_size, version = struct.unpack("{}2I".format(e), f.read(8))
            toc = [read_offset(f) for _ in range(7)]
            base_name = shpb_file[:-7]
            shps_file = base_name + '.TOSHPS' # vert
            shpp_file = base_name + '.TOSHPP' # uv/idx
            if not os.path.exists(shps_file):
                input(os.path.basename(shps_file) + " is missing, press Enter to skip processing.")
                return False
            elif not os.path.exists(shpp_file):
                input(os.path.basename(shpp_file) + " is missing, press Enter to skip processing.")
                return False
            else:
                morph_target_data = read_section_2 (f, toc[1], shps_file)
                meshes, bone_palette_ids, mesh_blocks_info = read_section_1 (f, toc[0],
                    shps_file, shpp_file, morph_target_data)
            material_struct = read_section_3 (f, toc[2])
            # Find and incorporate an external skeleton
            skel_struct = read_skel_file(find_skeleton(bone_palette_ids))
            vgmap = {'bone_{}'.format(bone_palette_ids[i]):i for i in range(len(bone_palette_ids))}
            if all([y in [x['id'] for x in skel_struct] for y in bone_palette_ids]):
                skel_index = {skel_struct[i]['id']:i for i in range(len(skel_struct))}
                vgmap = {skel_struct[skel_index[bone_palette_ids[i]]]['name']:i for i in range(len(bone_palette_ids))}
            for i in range(len(mesh_blocks_info)):
                mesh_blocks_info[i]['vgmap'] = 0
            vgmaps = [vgmap]
            if write_raw_buffers == True:
                if os.path.exists(base_name) and (os.path.isdir(base_name)) and (overwrite == False):
                    if str(input(base_name + " folder exists! Overwrite? (y/N) ")).lower()[0:1] == 'y':
                        overwrite = True
                if (overwrite == True) or not os.path.exists(base_name):
                    if not os.path.exists(base_name):
                        os.mkdir(base_name)
                    for i in range(len(meshes)):
                        filename = '{0:02d}_{1}'.format(i, mesh_blocks_info[i]['name'])
                        write_fmt(meshes[i]['fmt'], '{0}/{1}.fmt'.format(base_name, filename))
                        write_ib(meshes[i]['ib'], '{0}/{1}.ib'.format(base_name, filename), meshes[i]['fmt'], '<')
                        write_vb(meshes[i]['vb'], '{0}/{1}.vb'.format(base_name, filename), meshes[i]['fmt'], '<')
                        open('{0}/{1}.vgmap'.format(base_name, filename), 'wb').write(json.dumps(vgmap,indent=4).encode())
                        if len(meshes[i]['morphs']) > 0:
                            morphed_meshes = make_morphed_meshes(meshes[i])
                            for j in range(len(morphed_meshes)):
                                filename = '{0:02d}_{1}_morphed_{2:02d}'.format(i, mesh_blocks_info[i]['name'], j)
                                write_fmt(morphed_meshes[j]['fmt'], '{0}/{1}_{2}.fmt'.format(base_name, filename,
                                    morphed_meshes[j]['name']))
                                write_ib(morphed_meshes[j]['ib'], '{0}/{1}_{2}.ib'.format(base_name, filename,
                                    morphed_meshes[j]['name']), meshes[i]['fmt'], '<')
                                write_vb(morphed_meshes[j]['vb'], '{0}/{1}_{2}.vb'.format(base_name, filename,
                                    morphed_meshes[j]['name']), meshes[i]['fmt'], '<')
                                open('{0}/{1}.vgmap'.format(base_name, filename), 'wb').write(json.dumps(vgmap,indent=4).encode())
                    mesh_struct = [{y:x[y] for y in x if not any(
                        ['offset' in y, 'num' in y])} for x in mesh_blocks_info]
                    for i in range(len(mesh_struct)):
                        mesh_struct[i]['material'] = material_struct[mesh_struct[i]['material']]['name']
                    mesh_struct = [{'id_referenceonly': i, **mesh_struct[i]} for i in range(len(mesh_struct))]
                    write_struct_to_json(mesh_struct, base_name + '/mesh_info')
                    write_struct_to_json(material_struct, base_name + '/material_info')
                    write_struct_to_json(skel_struct, base_name + '/skeleton_info')
            write_gltf(base_name, skel_struct, vgmaps, mesh_blocks_info, meshes, material_struct,\
                overwrite = overwrite, write_binary_gltf = write_binary_gltf)
    return True

def process_shpb_combined (shpb_files, overwrite = False, write_binary_gltf = True):
    skel_struct, meshes, bone_palettes, vgmaps, mesh_blocks_info, material_struct, = [], [], [], [], [], []
    for i in range(len(shpb_files)):
        print("Processing {}...".format(shpb_files[i]))
        with open(shpb_files[i], 'rb') as f:
            magic = f.read(4)
            if magic in [b'MPSM', b'MSPM']:
                set_endianness({b'MPSM': '<', b'MSPM': '>'}[magic])
                file_size, version = struct.unpack("{}2I".format(e), f.read(8))
                toc = [read_offset(f) for _ in range(7)]
                base_name = shpb_files[i][:-7]
                shps_file = base_name + '.TOSHPS' # vert
                shpp_file = base_name + '.TOSHPP' # uv/idx
                if not os.path.exists(shps_file):
                    input(os.path.basename(shps_file) + " is missing, press Enter to skip processing.")
                    continue
                elif not os.path.exists(shpp_file):
                    input(os.path.basename(shpp_file) + " is missing, press Enter to skip processing.")
                    continue
                else:
                    morph_target_data_i = read_section_2 (f, toc[1], shps_file)
                    meshes_i, bone_palette_ids_i, mesh_blocks_info_i = read_section_1 (f, toc[0],
                        shps_file, shpp_file, morph_target_data_i)
                material_struct_i = read_section_3 (f, toc[2])
                for i in range(len(mesh_blocks_info_i)):
                    mesh_blocks_info_i[i]['material'] = mesh_blocks_info_i[i]['material'] + len(material_struct)
                    mesh_blocks_info_i[i]['vgmap'] = len(bone_palettes)
                bone_palettes.append(bone_palette_ids_i)
                meshes.extend(meshes_i)
                mesh_blocks_info.extend(mesh_blocks_info_i)
                material_struct.extend(material_struct_i)
    bone_palette_ids = list(set([x for y in bone_palettes for x in y]))
    skel_struct = read_skel_file(find_skeleton(bone_palette_ids))
    for i in range(len(bone_palettes)):
        vgmap = {'bone_{}'.format(bone_palettes[i][j]):j for j in range(len(bone_palettes[i]))}
        if all([y in [x['id'] for x in skel_struct] for y in bone_palettes[i]]):
            skel_index = {skel_struct[j]['id']:j for j in range(len(skel_struct))}
            vgmap = {skel_struct[skel_index[bone_palettes[i][j]]]['name']:j for j in range(len(bone_palettes[i]))}
        vgmaps.append(vgmap)
    common_name = []
    for i in range(len(os.path.basename(shpb_files[0]))):
        if all([os.path.basename(x)[:i] == os.path.basename(shpb_files[0])[:i] for x in shpb_files]):
            common_name = os.path.basename(shpb_files[0])[:i]
        else:
            break
    if len(common_name) > 1:
        base_name = common_name[:-1] if common_name[-1] == '_' else common_name
    else:
        base_name = base_name + '_combined'
    write_gltf(base_name, skel_struct, vgmaps, mesh_blocks_info, meshes, material_struct,\
        overwrite = overwrite, write_binary_gltf = write_binary_gltf)
    return True

if __name__ == "__main__":
    # Set current directory
    if getattr(sys, 'frozen', False):
        os.chdir(os.path.dirname(sys.executable))
    else:
        os.chdir(os.path.abspath(os.path.dirname(__file__)))

    if len(sys.argv) > 1:
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('-t', '--textformat', help="Write gltf instead of glb", action="store_false")
        parser.add_argument('-d', '--dumprawbuffers', help="Write fmt/ib/vb/vgmap files in addition to glb", action="store_true")
        parser.add_argument('-o', '--overwrite', help="Overwrite existing files", action="store_true")
        parser.add_argument('shpb_file', help="Name of model file to process.")
        args = parser.parse_args()
        if os.path.exists(args.shpb_file[:-7]+'.TOSHPB') and args.shpb_file[-7:-1].upper() == '.TOSHP':
            process_shpb(args.shpb_file[:-7]+'.TOSHPB', overwrite = args.overwrite, \
                write_raw_buffers = args.dumprawbuffers, write_binary_gltf = args.textformat)
    else:
        shpb_files = glob.glob('*.TOSHPB')
        if combine_models_into_single_gltf == True:
            process_shpb_combined(shpb_files)
        else:
            for shpb_file in shpb_files:
                process_shpb(shpb_file)