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
    import struct, json, numpy, copy, glob, os, sys
    from lib_fmtibvb import *
except ModuleNotFoundError as e:
    print("Python module missing! {}".format(e.msg))
    input("Press Enter to abort.")
    raise

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
        if not ib_list[i] == -1:
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
                inv_mtx = [struct.unpack("{}16f".format(e), f.read(64)) for _ in range(unk0[0])] # Stored correctly
                abs_mtx = [struct.unpack("{}16f".format(e), f.read(64)) for _ in range(unk0[0])] # Stored transposed
                abs_mtx_flip = [numpy.array(abs_mtx[i]).reshape(4,4).flatten('F').tolist() for i in range(len(abs_mtx))] # Column major
                dat2 = [struct.unpack("{}I2HI2HI2HI2H".format(e), f.read(32)) for _ in range(unk0[0])] # all the numbers are the same except ID number
                dat3 = [struct.unpack("{}I2H".format(e), f.read(8)) for _ in range(unk0[2])] # are these root nodes?
                dat4 = list(struct.unpack("{}{}I".format(e, unk0[0]), f.read(unk0[0] * 4))) # list of bones, equal to first val of dat2
                dat5 = [struct.unpack("{}2h".format(e), f.read(4)) for _ in range(unk0[0])] # the second value is -1 unless there is an entry in dat3 - again, root node info?
                dat6 = [struct.unpack("{}2h3f".format(e), f.read(16)) for _ in range(unk0[0])] # first value is probably parent, second value is 0 if in dat3, 1 if not
                # We need to skip the long list of floats, because I have no idea how to determine the length.  We will search for the first reasonably valid pointer
                val, = struct.unpack("{}I".format(e), f.read(4))
                while (val > 0xFFFF or val < 0x1):
                    val, = struct.unpack("{}I".format(e), f.read(4))
                f.seek(-4,1)
                names = []
                for _ in range(unk0[0]):
                    #3x u32 - start offset, end offset, blank
                    names.append(read_string(f, read_offset(f)))
                    f.seek(8,1) # skip end offset and blank
                skel_struct = [{'id': dat2[i][0], 'name': names[i], 'abs_matrix': abs_mtx_flip[i],\
                'inv_matrix': inv_mtx[i], 'parent': dat6[i][0]} for i in range(unk0[0])]
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
    for i in range(len(hrcb_files)):
        skel_struct = read_skel_file(hrcb_files[i])
        palettes[hrcb_files[i]] = [x['id'] for x in skel_struct]
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

def make_fmt(num_uvs, has_weights = True):
    fmt = {'stride': '0', 'topology': 'trianglelist', 'format':\
        "DXGI_FORMAT_R16_UINT", 'elements': []}
    element_id, stride = 0, 0
    semantic_index = {'TEXCOORD': 0} # Counters for multiple indicies
    elements = []
    for i in range(2 + num_uvs + (2 if has_weights else 0)):
            # I think order matters in this dict, so we will define the entire structure with default values
            element = {'id': '{0}'.format(element_id), 'SemanticName': '', 'SemanticIndex': '0',\
                'Format': '', 'InputSlot': '0', 'AlignedByteOffset': str(stride),\
                'InputSlotClass': 'per-vertex', 'InstanceDataStepRate': '0'}
            if i == 0:
                element['SemanticName'] = 'POSITION'
                element['Format'] = 'R32G32B32_FLOAT'
                stride += 12
            elif i == 1:
                element['SemanticName'] = 'NORMAL'
                element['Format'] = 'R32G32B32_FLOAT'
                stride += 12
            elif i == 4+num_uvs-2:
                element['SemanticName'] = 'BLENDWEIGHTS'
                element['Format'] = 'R32G32B32A32_FLOAT'
                stride += 16
            elif i == 4+num_uvs-1:
                element['SemanticName'] = 'BLENDINDICES'
                element['Format'] = 'R8G8B8A8_UINT'
                stride += 4
            else:
                element['SemanticName'] = 'TEXCOORD'
                element['SemanticIndex'] = str(semantic_index['TEXCOORD'])
                element['Format'] = 'R32G32_FLOAT'
                semantic_index['TEXCOORD'] += 1
                stride += 8
            element_id += 1
            elements.append(element)
    fmt['stride'] = str(stride)
    fmt['elements'] = elements
    return(fmt)

def read_mesh (mesh_info, vt_f, idx_f):
    def read_floats (f, num):
        return(list(struct.unpack("{0}{1}f".format(e, num), f.read(num * 4))))
    def read_bytes (f, num):
        return(list(struct.unpack("{0}{1}B".format(e, num), f.read(num))))
    def read_interleaved_floats (f, num, stride, total):
        vecs = []
        padding = stride - (num * 4)
        for i in range(total):
            vecs.append(read_floats(f, num))
            f.seek(padding, 1)
        return(vecs)
    def read_interleaved_bytes (f, num, stride, total):
        vecs = []
        padding = stride - (num)
        for i in range(total):
            vecs.append(read_bytes(f, num))
            f.seek(padding, 1)
        return(vecs)
    def fix_weights (weights):
        for _ in range(3):
            weights = [x+[round(1-sum(x),6)] if len(x) < 4 else x for x in weights]
        return(weights)
    vt_f.seek(mesh_info['vert_offset'])
    unk0, count, unk2 = struct.unpack("{}3I".format(e), vt_f.read(12))
    uv_stride = (8 * (mesh_info['flags'] & 0xF) + 4)
    num_uv_maps = mesh_info['flags'] & 0xF
    verts = []
    norms = []
    if mesh_info['flags'] & 0xF0 == 0x50:
        num_vertices_array = []
        blend_idx = []
        weights = []
        for i in range(count):
            num_vertices_array.append(list(struct.unpack("{}4H".format(e), vt_f.read(8))))
        for i in range(count):
            for j in range(len(num_vertices_array[i])):
                vert_offset = vt_f.tell()
                norm_offset = vert_offset + 12
                blend_idx_offset = norm_offset + 12
                weights_offset = blend_idx_offset + 4
                stride = 28 + (j * 4)
                end_offset = vt_f.tell() + (num_vertices_array[i][j] * stride)
                vt_f.seek(vert_offset)
                verts.extend(read_interleaved_floats(vt_f, 3, stride, num_vertices_array[i][j]))
                vt_f.seek(norm_offset)
                norms.extend(read_interleaved_floats(vt_f, 3, stride, num_vertices_array[i][j]))
                vt_f.seek(blend_idx_offset)
                blend_idx.extend(read_interleaved_bytes(vt_f, 4, stride, num_vertices_array[i][j]))
                if j > 0:
                    vt_f.seek(weights_offset)
                    weights.extend(read_interleaved_floats(vt_f, j, stride, num_vertices_array[i][j]))
                else:
                    weights.extend([[1.0] for _ in range(num_vertices_array[i][j])])
                vt_f.seek(end_offset)
            weights = fix_weights(weights)
    elif mesh_info['flags'] & 0xF0 == 0x70:
        unk_list = list(struct.unpack("<I2H", vt_f.read(8)))
        num_vertices = mesh_info['num_verts']
        vert_offset = vt_f.tell()
        norm_offset = vert_offset + 12
        stride = 24
        end_offset = vt_f.tell() + (num_vertices * stride)
        vt_f.seek(vert_offset)
        verts.extend(read_interleaved_floats(vt_f, 3, stride, num_vertices))
        vt_f.seek(norm_offset)
        norms.extend(read_interleaved_floats(vt_f, 3, stride, num_vertices))
        vt_f.seek(end_offset) # More data after this
    elif mesh_info['flags'] & 0xF0 == 0x0:
        print("FOUND 0x0!")
        num_vertices = mesh_info['num_verts']
        vert_offset = idx_f.seek(mesh_info['uv_offset'])
        norm_offset = vert_offset + 12
        stride = 28 + ((mesh_info['flags'] & 0xF) * 8) # 12 + 12 + 4 extra for UV padding
        idx_f.seek(vert_offset)
        verts.extend(read_interleaved_floats(idx_f, 3, stride, num_vertices))
        idx_f.seek(norm_offset)
        norms.extend(read_interleaved_floats(idx_f, 3, stride, num_vertices))
        vt_f.seek(20,1)
    uv_maps = []
    if not mesh_info['flags'] & 0xF0 == 0x0:
        for i in range(num_uv_maps):
            idx_f.seek(mesh_info['uv_offset'] + 4 + (i * 8))
            uv_maps.append(read_interleaved_floats (idx_f, 2, uv_stride, mesh_info['num_verts']))
    else:
        for i in range(num_uv_maps):
            idx_f.seek(mesh_info['uv_offset'] + 28)
            uv_maps.append(read_interleaved_floats (idx_f, 2, stride, mesh_info['num_verts']))
    idx_f.seek(mesh_info['idx_offset'])
    idx_buffer = list(struct.unpack("{0}{1}h".format(e, mesh_info['num_idx']), idx_f.read(mesh_info['num_idx'] * 2)))
    fmt = make_fmt(len(uv_maps), True)
    vb = [{'Buffer': verts}, {'Buffer': norms}]
    for uv_map in uv_maps:
        vb.append({'Buffer': uv_map})
    if mesh_info['flags'] & 0xF0 == 0x50:
        vb.append({'Buffer': weights})
        vb.append({'Buffer': blend_idx})
    elif mesh_info['flags'] & 0xF0 in [0x0, 0x70]:
        vb.append({'Buffer': [[1.0, 0.0, 0.0, 0.0] for _ in range(len(verts))]})
        vb.append({'Buffer': [[0, 0, 0, 0] for _ in range(len(verts))]})
    return({'fmt': fmt, 'vb': vb, 'ib': trianglestrip_to_list(idx_buffer)})

#Meshes, offset should be toc[1].  Requires shps filename for verts, and shpp filename for uv's and index buffer.
def read_section_1 (f, offset, shps_file, shpp_file):
    f.seek(offset)
    mesh_count, palette_count = struct.unpack("{}2H".format(e), f.read(4))
    set_0 = [struct.unpack("{}4f".format(e), f.read(16)) for _ in range(mesh_count)]
    set_1 = []
    names = []
    for _ in range(mesh_count):
        set_1.append(struct.unpack("{}6H".format(e), f.read(12)))
        names.append(read_string(f, read_offset (f)))
    mesh_blocks_info = []
    for i in range(mesh_count):
        mesh_info = {'name': names[i]}
        mesh_info['num_verts'], mesh_info['num_idx'], mesh_info['uv_offset'], mesh_info['idx_offset'], mesh_info['vert_offset'] = \
            struct.unpack("{}2H3I".format(e), f.read(16))
        mesh_info['unk0'] = set_1[i][0]
        mesh_info['mesh'] = set_1[i][1]
        mesh_info['submesh'] = set_1[i][2]
        mesh_info['flags'] = set_1[i][3]
        mesh_info['material'] = set_1[i][4]
        mesh_info['unk1'] = set_1[i][5]
        mesh_info['unk_floats'] = set_0[i]
        mesh_blocks_info.append(mesh_info)
    bone_palette_ids = struct.unpack("{}{}I".format(e, palette_count), f.read(4 * palette_count))
    meshes = []
    with open (shps_file, 'rb') as vt_f:
        with open (shpp_file, 'rb') as idx_f:
            for i in range(mesh_count):
                meshes.append(read_mesh(mesh_blocks_info[i], vt_f, idx_f))
    return(meshes, bone_palette_ids, mesh_blocks_info)

#Materials, offset should be toc[1]. 
def read_section_3 (f, offset):
    f.seek(offset)
    counts = struct.unpack("{}8H".format(e), f.read(16))
    mat_count = counts[0]
    tex_pointer_count = counts[1]
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
    if counts[2] > 0:
        print("Unreversed material block detected, attempting to skip past...")
        val, = struct.unpack("{}I".format(e), f.read(4))
        while (val > 0xFFFF or val < 0x1):
            val, = struct.unpack("{}I".format(e), f.read(4))
        f.seek(-4,1)
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
    new_semantics = {'BLENDWEIGHTS': 'WEIGHTS', 'BLENDINDICES': 'JOINTS'}
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

def fix_strides(submesh):
    offset = 0
    for i in range(len(submesh['vb'])):
        submesh['vb'][i]['fmt']['AlignedByteOffset'] = str(offset)
        submesh['vb'][i]['stride'] = get_stride_from_dxgi_format(submesh['vb'][i]['fmt']['Format'])
        offset += submesh['vb'][i]['stride']
    return(submesh)

def write_gltf(base_name, skel_struct, vgmap, mesh_blocks_info, meshes, material_struct,\
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
    gltf_data['scenes'][0]['nodes'] = [0]
    gltf_data['scene'] = 0
    gltf_data['skins'] = []
    gltf_data['textures'] = []
    giant_buffer = bytes()
    buffer_view = 0
    # Materials
    material_dict = [{'name': material_struct[i]['name'], 'texture': material_struct[i]['textures'][0],
        'alpha': material_struct[i]['alpha'] if 'alpha' in material_struct[i] else 0}
        for i in range(len(material_struct))]
    texture_list = sorted(list(set([x['texture'] for x in material_dict])))
    gltf_data['images'] = [{'uri':'textures/{}.dds'.format(x)} for x in texture_list]
    for mat in material_dict:
        material = { 'name': mat['name'] }
        sampler = { 'wrapS': 10497, 'wrapT': 10497 } # I have no idea if this setting exists
        texture = { 'source': texture_list.index(mat['texture']), 'sampler': len(gltf_data['samplers']) }
        material['pbrMetallicRoughness']= { 'baseColorTexture' : { 'index' : len(gltf_data['textures']), },\
            'metallicFactor' : 0.0, 'roughnessFactor' : 1.0 }
        if mat['alpha'] & 1:
            material['alphaMode'] = 'MASK'
        gltf_data['samplers'].append(sampler)
        gltf_data['textures'].append(texture)
        gltf_data['materials'].append(material)
    material_list = [x['name'] for x in gltf_data['materials']]
    missing_textures = [x['uri'] for x in gltf_data['images'] if not os.path.exists(x['uri'])]
    if len(missing_textures) > 0:
        print("Warning:  The following textures were not found:")
        for texture in missing_textures:
            print("{}".format(texture))
    # Nodes
    for i in range(len(skel_struct)):
        g_node = {'children': skel_struct[i]['children'], 'name': skel_struct[i]['name'], 'matrix': skel_struct[i]['matrix']}
        gltf_data['nodes'].append(g_node)
    for i in range(len(gltf_data['nodes'])):
        if len(gltf_data['nodes'][i]['children']) == 0:
            del(gltf_data['nodes'][i]['children'])
    if len(gltf_data['nodes']) == 0:
        gltf_data['nodes'].append({'children': [], 'name': 'root'})
    # Mesh nodes will be attached to the first node since in the original model, they don't really have a home
    node_id_list = [x['id'] for x in skel_struct]
    mesh_node_ids = {x['mesh']:x['name'] for x in mesh_blocks_info}
    for mesh_node_id in mesh_node_ids:
        if not mesh_node_id in node_id_list:
            g_node = {'name': mesh_node_ids[mesh_node_id]}
            gltf_data['nodes'][0]['children'].append(len(gltf_data['nodes']))
            gltf_data['nodes'].append(g_node)
    mesh_block_tree = {x:[i for i in range(len(mesh_blocks_info)) if mesh_blocks_info[i]['mesh'] == x] for x in mesh_node_ids}
    node_list = [x['name'] for x in gltf_data['nodes']]
    # Skin matrices
    skinning_possible = True
    try:
        vgmap_nodes = [node_list.index(x) for x in list(vgmap.keys())]
        ibms = [skel_struct[j]['inv_matrix'] for j in vgmap_nodes]
        inv_mtx_buffer = b''.join([struct.pack("<16f", *x) for x in ibms])
    except ValueError:
        skinning_possible = False
    # Meshes
    mesh_names = [] # Xillia doesn't have a 
    for mesh in mesh_block_tree: #Mesh
        primitives = []
        for j in range(len(mesh_block_tree[mesh])): #Submesh
            i = mesh_block_tree[mesh][j]
            # Vertex Buffer
            gltf_fmt = convert_fmt_for_gltf(meshes[i]['fmt'])
            vb_stream = io.BytesIO()
            write_vb_stream(meshes[i]['vb'], vb_stream, gltf_fmt, e='<', interleave = False)
            block_offset = len(giant_buffer)
            primitive = {"attributes":{}}
            for element in range(len(gltf_fmt['elements'])):
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
            primitives.append(primitive)
        if len(primitives) > 0:
            if mesh_node_ids[mesh] in node_list: # One of the new nodes
                node_id = node_list.index(mesh_node_ids[mesh])
            else: # One of the pre-assigned nodes
                node_id = node_id_list.index(mesh_blocks_info[i]["mesh"])
            gltf_data['nodes'][node_id]['mesh'] = len(gltf_data['meshes'])
            gltf_data['meshes'].append({"primitives": primitives, "name": mesh_node_ids[mesh]})
            # Skinning
            if len(vgmap) > 0 and skinning_possible == True:
                gltf_data['nodes'][node_id]['skin'] = len(gltf_data['skins'])
                gltf_data['skins'].append({"inverseBindMatrices": len(gltf_data['accessors']),\
                    "joints": [node_list.index(x) for x in vgmap]})
                gltf_data['accessors'].append({"bufferView" : len(gltf_data['bufferViews']),\
                    "componentType": 5126,\
                    "count": len(ibms),\
                    "type": "MAT4"})
                gltf_data['bufferViews'].append({"buffer": 0,\
                    "byteOffset": len(giant_buffer),\
                    "byteLength": len(inv_mtx_buffer)})
                giant_buffer += inv_mtx_buffer
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
            unk_int, = struct.unpack("{}I".format(e), f.read(4))
            toc = [read_offset(f) for _ in range(5)]
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
                meshes, bone_palette_ids, mesh_blocks_info = read_section_1 (f, toc[1], shps_file, shpp_file)
            material_struct = read_section_3 (f, toc[3])
            # Find and incorporate an external skeleton
            skel_struct = read_skel_file(find_skeleton(bone_palette_ids))
            vgmap = {'bone_{}'.format(bone_palette_ids[i]):i for i in range(len(bone_palette_ids))}
            if all([y in [x['id'] for x in skel_struct] for y in bone_palette_ids]):
                skel_index = {skel_struct[i]['id']:i for i in range(len(skel_struct))}
                vgmap = {skel_struct[skel_index[bone_palette_ids[i]]]['name']:i for i in range(len(bone_palette_ids))}
            write_gltf(base_name, skel_struct, vgmap, mesh_blocks_info, meshes, material_struct,\
                overwrite = overwrite, write_binary_gltf = write_binary_gltf)
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
                    mesh_struct = [{y:x[y] for y in x if not any(
                        ['offset' in y, 'num' in y])} for x in mesh_blocks_info]
                    for i in range(len(mesh_struct)):
                        mesh_struct[i]['material'] = material_struct[mesh_struct[i]['material']]['name']
                    mesh_struct = [{'id_referenceonly': i, **mesh_struct[i]} for i in range(len(mesh_struct))]
                    write_struct_to_json(mesh_struct, base_name + '/mesh_info')
                    write_struct_to_json(material_struct, base_name + '/material_info')
                    #write_struct_to_json(skel_struct, base_name + '/skeleton_info')
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
        for shpb_file in shpb_files:
            process_shpb(shpb_file)
