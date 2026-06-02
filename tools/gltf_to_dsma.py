#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
#
# Copyright (c) 2026 Affinity contributors
#
# glTF / GLB front-end for the DSMA converter. It produces exactly the same DSM
# (geometry) and DSA (animation) files as md5_to_dsma, but reads modern glTF
# instead of MD5. Only the parsing differs — the DS display-list generation and
# the DSA fixed-point layout are shared via dsma_common.
#
# Constraints inherited from DSMA itself (not from glTF):
#   - Rigid skinning only: each vertex is bound to a single bone. Multi-weight
#     glTF vertices are collapsed to their dominant bone (a warning is printed).
#   - At most 29 bones (each bone occupies one DS matrix-stack slot).
#   - No per-bone scale (the DSA stores translation + quaternion only). Bone
#     scale far from 1.0 triggers a warning.
#
# This is a zero-dependency tool: only the Python standard library is used.

import base64
import json
import os
import struct
from math import sqrt

from dsma_common import (Vector, Quaternion, Joint, save_animation,
                         emit_triangles_to_dsm)


class GLTFFormatError(Exception):
    pass


VALID_TEXTURE_SIZES = [8, 16, 32, 64, 128, 256, 512, 1024]

# componentType -> (struct char, byte size)
_COMPONENT = {
    5120: ('b', 1),  # BYTE
    5121: ('B', 1),  # UNSIGNED_BYTE
    5122: ('h', 2),  # SHORT
    5123: ('H', 2),  # UNSIGNED_SHORT
    5125: ('I', 4),  # UNSIGNED_INT
    5126: ('f', 4),  # FLOAT
}

# accessor type -> number of components
_NUM_COMPONENTS = {
    "SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4,
    "MAT2": 4, "MAT3": 9, "MAT4": 16,
}


# ---------------------------------------------------------------------------
# glTF / GLB container loading
# ---------------------------------------------------------------------------

def load_gltf(path):
    """Return (gltf_json_dict, [bytes_per_buffer]) for a .gltf or .glb file."""
    with open(path, "rb") as f:
        data = f.read()

    if data[:4] == b'glTF':
        # Binary glTF container.
        magic, version, length = struct.unpack_from("<III", data, 0)
        if version != 2:
            raise GLTFFormatError(f"Unsupported GLB version: {version}")
        gltf = None
        bin_chunk = None
        offset = 12
        while offset < length:
            chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
            offset += 8
            chunk = data[offset:offset + chunk_len]
            offset += chunk_len
            if chunk_type == 0x4E4F534A:      # 'JSON'
                gltf = json.loads(chunk.decode("utf-8"))
            elif chunk_type == 0x004E4942:    # 'BIN\0'
                bin_chunk = chunk
        if gltf is None:
            raise GLTFFormatError("GLB has no JSON chunk")
        buffers = _resolve_buffers(gltf, os.path.dirname(path), bin_chunk)
        return gltf, buffers

    # Text glTF.
    gltf = json.loads(data.decode("utf-8"))
    buffers = _resolve_buffers(gltf, os.path.dirname(path), None)
    return gltf, buffers


def _resolve_buffers(gltf, base_dir, glb_bin):
    buffers = []
    for buf in gltf.get("buffers", []):
        uri = buf.get("uri")
        if uri is None:
            # Buffer backed by the GLB BIN chunk.
            if glb_bin is None:
                raise GLTFFormatError("Buffer without uri but no GLB BIN chunk")
            buffers.append(glb_bin)
        elif uri.startswith("data:"):
            # Embedded base64 data URI.
            comma = uri.index(",")
            buffers.append(base64.b64decode(uri[comma + 1:]))
        else:
            # External .bin file (URI is percent-encoded relative path).
            from urllib.parse import unquote
            with open(os.path.join(base_dir, unquote(uri)), "rb") as bf:
                buffers.append(bf.read())
    return buffers


def read_accessor(gltf, buffers, accessor_index):
    """
    Decode an accessor into a list of elements. Each element is a tuple of
    floats (or a single float for SCALAR). Honours bufferView byteStride and the
    accessor's `normalized` flag.
    """
    acc = gltf["accessors"][accessor_index]
    comp_char, comp_size = _COMPONENT[acc["componentType"]]
    ncomp = _NUM_COMPONENTS[acc["type"]]
    count = acc["count"]
    normalized = acc.get("normalized", False)

    view = gltf["bufferViews"][acc["bufferView"]]
    buf = buffers[view["buffer"]]
    base = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = view.get("byteStride", comp_size * ncomp)

    # Normalization divisor for integer component types.
    norm_div = {5121: 255.0, 5123: 65535.0, 5120: 127.0, 5122: 32767.0}

    out = []
    for i in range(count):
        elem_off = base + i * stride
        vals = struct.unpack_from("<" + comp_char * ncomp, buf, elem_off)
        if normalized and acc["componentType"] in norm_div:
            d = norm_div[acc["componentType"]]
            if acc["componentType"] in (5120, 5122):  # signed
                vals = tuple(max(v / d, -1.0) for v in vals)
            else:
                vals = tuple(v / d for v in vals)
        out.append(vals[0] if ncomp == 1 else vals)
    return out


# ---------------------------------------------------------------------------
# Minimal 4x4 matrix / quaternion math (row-major, point = M * v)
# ---------------------------------------------------------------------------

def mat4_identity():
    return [1.0, 0, 0, 0,  0, 1.0, 0, 0,  0, 0, 1.0, 0,  0, 0, 0, 1.0]


def mat4_from_gltf(col_major):
    """glTF stores matrices column-major; convert to our row-major layout."""
    m = [0.0] * 16
    for col in range(4):
        for row in range(4):
            m[row * 4 + col] = col_major[col * 4 + row]
    return m


def mat4_mul(a, b):
    out = [0.0] * 16
    for r in range(4):
        for c in range(4):
            out[r * 4 + c] = (a[r * 4 + 0] * b[0 * 4 + c] +
                              a[r * 4 + 1] * b[1 * 4 + c] +
                              a[r * 4 + 2] * b[2 * 4 + c] +
                              a[r * 4 + 3] * b[3 * 4 + c])
    return out


def mat4_mul_point(m, x, y, z):
    px = m[0] * x + m[1] * y + m[2] * z + m[3]
    py = m[4] * x + m[5] * y + m[6] * z + m[7]
    pz = m[8] * x + m[9] * y + m[10] * z + m[11]
    return Vector(px, py, pz)


def quat_to_mat4(qx, qy, qz, qw):
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return [1 - 2 * (yy + zz),     2 * (xy - wz),     2 * (xz + wy), 0,
                2 * (xy + wz), 1 - 2 * (xx + zz),     2 * (yz - wx), 0,
                2 * (xz - wy),     2 * (yz + wx), 1 - 2 * (xx + yy), 0,
                            0,                 0,                 0, 1]


def mat4_from_trs(t, q, s):
    """t=(x,y,z), q=(x,y,z,w), s=(x,y,z). Returns T*R*S (row-major)."""
    m = quat_to_mat4(q[0], q[1], q[2], q[3])
    # Scale the rotation columns.
    for r in range(3):
        m[r * 4 + 0] *= s[0]
        m[r * 4 + 1] *= s[1]
        m[r * 4 + 2] *= s[2]
    # Translation.
    m[3], m[7], m[11] = t[0], t[1], t[2]
    return m


def mat4_invert(m):
    """General 4x4 inverse (cofactor method)."""
    inv = [0.0] * 16
    inv[0] = (m[5]*m[10]*m[15] - m[5]*m[11]*m[14] - m[9]*m[6]*m[15] +
              m[9]*m[7]*m[14] + m[13]*m[6]*m[11] - m[13]*m[7]*m[10])
    inv[4] = (-m[4]*m[10]*m[15] + m[4]*m[11]*m[14] + m[8]*m[6]*m[15] -
              m[8]*m[7]*m[14] - m[12]*m[6]*m[11] + m[12]*m[7]*m[10])
    inv[8] = (m[4]*m[9]*m[15] - m[4]*m[11]*m[13] - m[8]*m[5]*m[15] +
              m[8]*m[7]*m[13] + m[12]*m[5]*m[11] - m[12]*m[7]*m[9])
    inv[12] = (-m[4]*m[9]*m[14] + m[4]*m[10]*m[13] + m[8]*m[5]*m[14] -
               m[8]*m[6]*m[13] - m[12]*m[5]*m[10] + m[12]*m[6]*m[9])
    inv[1] = (-m[1]*m[10]*m[15] + m[1]*m[11]*m[14] + m[9]*m[2]*m[15] -
              m[9]*m[3]*m[14] - m[13]*m[2]*m[11] + m[13]*m[3]*m[10])
    inv[5] = (m[0]*m[10]*m[15] - m[0]*m[11]*m[14] - m[8]*m[2]*m[15] +
              m[8]*m[3]*m[14] + m[12]*m[2]*m[11] - m[12]*m[3]*m[10])
    inv[9] = (-m[0]*m[9]*m[15] + m[0]*m[11]*m[13] + m[8]*m[1]*m[15] -
              m[8]*m[3]*m[13] - m[12]*m[1]*m[11] + m[12]*m[3]*m[9])
    inv[13] = (m[0]*m[9]*m[14] - m[0]*m[10]*m[13] - m[8]*m[1]*m[14] +
               m[8]*m[2]*m[13] + m[12]*m[1]*m[10] - m[12]*m[2]*m[9])
    inv[2] = (m[1]*m[6]*m[15] - m[1]*m[7]*m[14] - m[5]*m[2]*m[15] +
              m[5]*m[3]*m[14] + m[13]*m[2]*m[7] - m[13]*m[3]*m[6])
    inv[6] = (-m[0]*m[6]*m[15] + m[0]*m[7]*m[14] + m[4]*m[2]*m[15] -
              m[4]*m[3]*m[14] - m[12]*m[2]*m[7] + m[12]*m[3]*m[6])
    inv[10] = (m[0]*m[5]*m[15] - m[0]*m[7]*m[13] - m[4]*m[1]*m[15] +
               m[4]*m[3]*m[13] + m[12]*m[1]*m[7] - m[12]*m[3]*m[5])
    inv[14] = (-m[0]*m[5]*m[14] + m[0]*m[6]*m[13] + m[4]*m[1]*m[14] -
               m[4]*m[2]*m[13] - m[12]*m[1]*m[6] + m[12]*m[2]*m[5])
    inv[3] = (-m[1]*m[6]*m[11] + m[1]*m[7]*m[10] + m[5]*m[2]*m[11] -
              m[5]*m[3]*m[10] - m[9]*m[2]*m[7] + m[9]*m[3]*m[6])
    inv[7] = (m[0]*m[6]*m[11] - m[0]*m[7]*m[10] - m[4]*m[2]*m[11] +
              m[4]*m[3]*m[10] + m[8]*m[2]*m[7] - m[8]*m[3]*m[6])
    inv[11] = (-m[0]*m[5]*m[11] + m[0]*m[7]*m[9] + m[4]*m[1]*m[11] -
               m[4]*m[3]*m[9] - m[8]*m[1]*m[7] + m[8]*m[3]*m[5])
    inv[15] = (m[0]*m[5]*m[10] - m[0]*m[6]*m[9] - m[4]*m[1]*m[10] +
               m[4]*m[2]*m[9] + m[8]*m[1]*m[6] - m[8]*m[2]*m[5])

    det = m[0]*inv[0] + m[1]*inv[4] + m[2]*inv[8] + m[3]*inv[12]
    if abs(det) < 1e-20:
        raise GLTFFormatError("Singular matrix cannot be inverted")
    det = 1.0 / det
    return [v * det for v in inv]


def mat4_decompose(m, warn_scale_name=None):
    """Return (Vector translation, Quaternion orient). Asserts ~unit scale."""
    t = Vector(m[3], m[7], m[11])

    # Column vectors of the upper-left 3x3.
    cx = (m[0], m[4], m[8])
    cy = (m[1], m[5], m[9])
    cz = (m[2], m[6], m[10])
    sx = sqrt(cx[0]**2 + cx[1]**2 + cx[2]**2)
    sy = sqrt(cy[0]**2 + cy[1]**2 + cy[2]**2)
    sz = sqrt(cz[0]**2 + cz[1]**2 + cz[2]**2)

    if warn_scale_name is not None:
        for s in (sx, sy, sz):
            if abs(s - 1.0) > 0.01:
                print(f"  WARNING: bone '{warn_scale_name}' has scale {sx:.3f},"
                      f"{sy:.3f},{sz:.3f}; DSMA ignores bone scale.")
                break

    sx = sx or 1.0
    sy = sy or 1.0
    sz = sz or 1.0
    # Normalized rotation matrix (row-major 3x3 embedded as r[row][col]).
    r = [[m[0] / sx, m[1] / sy, m[2] / sz],
         [m[4] / sx, m[5] / sy, m[6] / sz],
         [m[8] / sx, m[9] / sy, m[10] / sz]]

    # Matrix -> quaternion (Shepperd's method).
    trace = r[0][0] + r[1][1] + r[2][2]
    if trace > 0:
        sca = sqrt(trace + 1.0) * 2
        qw = 0.25 * sca
        qx = (r[2][1] - r[1][2]) / sca
        qy = (r[0][2] - r[2][0]) / sca
        qz = (r[1][0] - r[0][1]) / sca
    elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        sca = sqrt(1.0 + r[0][0] - r[1][1] - r[2][2]) * 2
        qw = (r[2][1] - r[1][2]) / sca
        qx = 0.25 * sca
        qy = (r[0][1] + r[1][0]) / sca
        qz = (r[0][2] + r[2][0]) / sca
    elif r[1][1] > r[2][2]:
        sca = sqrt(1.0 + r[1][1] - r[0][0] - r[2][2]) * 2
        qw = (r[0][2] - r[2][0]) / sca
        qx = (r[0][1] + r[1][0]) / sca
        qy = 0.25 * sca
        qz = (r[1][2] + r[2][1]) / sca
    else:
        sca = sqrt(1.0 + r[2][2] - r[0][0] - r[1][1]) * 2
        qw = (r[1][0] - r[0][1]) / sca
        qx = (r[0][2] + r[2][0]) / sca
        qy = (r[1][2] + r[2][1]) / sca
        qz = 0.25 * sca

    return t, Quaternion(qw, qx, qy, qz).normalize()


# ---------------------------------------------------------------------------
# Node hierarchy
# ---------------------------------------------------------------------------

def build_parent_map(gltf):
    parent = {}
    for i, node in enumerate(gltf.get("nodes", [])):
        for c in node.get("children", []):
            parent[c] = i
    return parent


def node_local_matrix(node, trs_override=None):
    """Local matrix of a node, optionally overriding T/R/S from animation."""
    if "matrix" in node and trs_override is None:
        return mat4_from_gltf(node["matrix"])
    t = node.get("translation", [0, 0, 0])
    r = node.get("rotation", [0, 0, 0, 1])
    s = node.get("scale", [1, 1, 1])
    if trs_override is not None:
        t = trs_override.get("t", t)
        r = trs_override.get("r", r)
        s = trs_override.get("s", s)
    return mat4_from_trs(t, r, s)


def global_matrix(node_index, parent_map, local_cache):
    """Compose a node's global matrix from cached local matrices."""
    chain = []
    n = node_index
    while n is not None:
        chain.append(n)
        n = parent_map.get(n)
    m = mat4_identity()
    for n in reversed(chain):
        m = mat4_mul(m, local_cache[n])
    return m


# ---------------------------------------------------------------------------
# Animation sampling
# ---------------------------------------------------------------------------

def sample_quat(times, values, t):
    k = _find_segment(times, t)
    if k is None:
        q = values[0]
        return [q[0], q[1], q[2], q[3]]
    if k + 1 >= len(times):
        q = values[-1]
        return [q[0], q[1], q[2], q[3]]
    t0, t1 = times[k], times[k + 1]
    a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
    q0, q1 = values[k], values[k + 1]
    # Hemisphere correction, then normalized lerp.
    dot = sum(q0[i] * q1[i] for i in range(4))
    sign = -1.0 if dot < 0 else 1.0
    q = [q0[i] * (1 - a) + sign * q1[i] * a for i in range(4)]
    mag = sqrt(sum(c * c for c in q)) or 1.0
    return [c / mag for c in q]


def sample_vec3(times, values, t):
    k = _find_segment(times, t)
    if k is None:
        return list(values[0])
    if k + 1 >= len(times):
        return list(values[-1])
    t0, t1 = times[k], times[k + 1]
    a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
    v0, v1 = values[k], values[k + 1]
    return [v0[i] * (1 - a) + v1[i] * a for i in range(3)]


def _find_segment(times, t):
    if t <= times[0]:
        return None if t < times[0] else 0
    for k in range(len(times) - 1):
        if times[k] <= t <= times[k + 1]:
            return k
    return len(times) - 1


def gather_animation(gltf, buffers, anim):
    """
    Return (sorted_frame_times, channels) where channels maps
    node_index -> {'t': (times, vals), 'r': (...), 's': (...)} of sampled data.
    """
    samplers = anim["samplers"]
    channels = {}
    all_times = set()

    for ch in anim["channels"]:
        target = ch["target"]
        node = target.get("node")
        path = target["path"]
        if node is None or path == "weights":
            continue  # morph targets unsupported
        smp = samplers[ch["sampler"]]
        interp = smp.get("interpolation", "LINEAR")
        times = read_accessor(gltf, buffers, smp["input"])
        vals = read_accessor(gltf, buffers, smp["output"])
        if interp == "CUBICSPLINE":
            # Keep only the keyframe value (drop in/out tangents).
            vals = vals[1::3]
            print(f"  WARNING: CUBICSPLINE on node {node}/{path} approximated as linear.")
        times = [float(x) for x in times]
        all_times.update(times)
        key = {"translation": "t", "rotation": "r", "scale": "s"}[path]
        channels.setdefault(node, {})[key] = (times, vals)

    return sorted(all_times), channels


# ---------------------------------------------------------------------------
# Top-level conversion
# ---------------------------------------------------------------------------

def find_skinned_mesh(gltf):
    """Return (node_index, mesh_index, skin_index) of the first skinned mesh."""
    for i, node in enumerate(gltf.get("nodes", [])):
        if "mesh" in node and "skin" in node:
            return i, node["mesh"], node["skin"]
    raise GLTFFormatError("No skinned mesh (a node with both 'mesh' and 'skin') found")


def convert(gltf_path, name, output_folder, texture_size, extension_mesh,
            extension_anim, export_base_pose, draw_normal_polygons, flip_winding):

    print(f"Loading glTF: {gltf_path}")
    gltf, buffers = load_gltf(gltf_path)

    mesh_node_idx, mesh_idx, skin_idx = find_skinned_mesh(gltf)
    skin = gltf["skins"][skin_idx]
    joint_nodes = skin["joints"]
    num_joints = len(joint_nodes)
    print(f"Loaded skin with {num_joints} joint(s).")

    if num_joints > 29:
        raise GLTFFormatError(
            f"{num_joints} bones exceeds the DS matrix-stack limit of 29.")

    # node index -> joint index (0..num_joints-1)
    node_to_joint = {n: j for j, n in enumerate(joint_nodes)}

    parent_map = build_parent_map(gltf)

    # Inverse bind matrices (one per joint, in skin.joints order).
    ibm_raw = read_accessor(gltf, buffers, skin["inverseBindMatrices"])
    ibms = [mat4_from_gltf(list(m)) for m in ibm_raw]

    # Bind-pose absolute joint transforms = inverse(IBM). These feed the DSM
    # (normals) and the optional base-pose DSA.
    bind_joints = []
    for j in range(num_joints):
        gmat = mat4_invert(ibms[j])
        node_name = gltf["nodes"][joint_nodes[j]].get("name", f"joint{j}")
        t, q = mat4_decompose(gmat, warn_scale_name=node_name)
        parent_node = parent_map.get(joint_nodes[j])
        parent_joint = node_to_joint.get(parent_node, -1)
        bind_joints.append(Joint(node_name, parent_joint, t, q))

    # ---- Geometry: build the neutral triangle list -------------------------
    mesh = gltf["meshes"][mesh_idx]
    triangles = []
    multi_weight_warned = False

    for prim in mesh["primitives"]:
        attr = prim["attributes"]
        positions = read_accessor(gltf, buffers, attr["POSITION"])
        texcoords = read_accessor(gltf, buffers, attr["TEXCOORD_0"])
        joints0 = read_accessor(gltf, buffers, attr["JOINTS_0"])
        weights0 = read_accessor(gltf, buffers, attr["WEIGHTS_0"])

        if "indices" in prim:
            indices = [int(i) for i in read_accessor(gltf, buffers, prim["indices"])]
        else:
            indices = list(range(len(positions)))

        # Resolve each vertex to a single (joint_index, joint-space pos, st).
        verts = []
        for vi in range(len(positions)):
            jw = list(zip(joints0[vi], weights0[vi]))
            jw.sort(key=lambda x: x[1], reverse=True)
            best_joint_local, best_weight = jw[0]
            if not multi_weight_warned and best_weight < 0.999:
                print("  WARNING: multi-weight vertices found; collapsing each to "
                      "its dominant bone (rigid skinning).")
                multi_weight_warned = True
            joint_index = int(best_joint_local)

            px, py, pz = positions[vi][0], positions[vi][1], positions[vi][2]
            # Vertex into joint-local space: IBM_joint * position.
            jpos = mat4_mul_point(ibms[joint_index], px, py, pz)
            st = (texcoords[vi][0], texcoords[vi][1])
            verts.append((joint_index, jpos, st))

        for k in range(0, len(indices), 3):
            tri = [verts[indices[k]], verts[indices[k + 1]], verts[indices[k + 2]]]
            if flip_winding:
                tri = [tri[2], tri[1], tri[0]]
            triangles.append(tri)

    print(f"  Triangles: {len(triangles)}")

    if export_base_pose:
        print("Converting base pose...")
        save_animation([bind_joints],
                       os.path.join(output_folder, f"{name}{extension_anim}"))

    print("Generating display list...")
    emit_triangles_to_dsm(bind_joints, triangles, texture_size,
                          os.path.join(output_folder, f"{name}{extension_mesh}"),
                          draw_normal_polygons)

    # ---- Animations --------------------------------------------------------
    for anim in gltf.get("animations", []):
        anim_name = anim.get("name", f"anim{gltf['animations'].index(anim)}")
        anim_name = anim_name.replace(".md5anim", "").replace(".", "_").lower()
        print(f"Converting animation: {anim.get('name', anim_name)}")

        frame_times, channels = gather_animation(gltf, buffers, anim)
        if not frame_times:
            print("  (no keyframes; skipped)")
            continue

        frames = []
        for t in frame_times:
            # Local matrix of every node at time t (animated channels override).
            local_cache = {}
            for ni, node in enumerate(gltf.get("nodes", [])):
                ch = channels.get(ni)
                override = None
                if ch is not None:
                    override = {}
                    if "t" in ch:
                        override["t"] = sample_vec3(ch["t"][0], ch["t"][1], t)
                    if "r" in ch:
                        override["r"] = sample_quat(ch["r"][0], ch["r"][1], t)
                    if "s" in ch:
                        override["s"] = sample_vec3(ch["s"][0], ch["s"][1], t)
                local_cache[ni] = node_local_matrix(node, override)

            frame_joints = []
            for j in range(num_joints):
                gmat = global_matrix(joint_nodes[j], parent_map, local_cache)
                pos, orient = mat4_decompose(gmat)
                frame_joints.append(Joint("", -1, pos, orient))
            frames.append(frame_joints)

        print(f"  Frames: {len(frames)}")
        save_animation(frames, os.path.join(output_folder,
                       f"{name}_{anim_name}{extension_anim}"))


if __name__ == "__main__":
    import argparse
    import sys
    import traceback

    print("gltf_to_dsma v0.1.0")
    print("")

    parser = argparse.ArgumentParser(
        description="Converts a skinned glTF/GLB model into DSM and DSA files.")
    parser.add_argument("--model", required=True,
                        help="input .gltf or .glb file")
    parser.add_argument("--name", required=True,
                        help="model name to be used in output files")
    parser.add_argument("--output", required=True, help="output folder")
    parser.add_argument("--texture", required=True, type=int, nargs="+",
                        action="extend", default=[],
                        help="texture width and height (e.g. '--texture 32 64')")
    parser.add_argument("--bin", action="store_true",
                        help="add '.bin' to the name of the output files")
    parser.add_argument("--export-base-pose", action="store_true",
                        help="export the bind pose as a one-frame DSA file")
    parser.add_argument("--no-flip-winding", action="store_true",
                        help="keep glTF triangle winding (default reverses it)")
    parser.add_argument("--draw-normal-polygons", action="store_true",
                        help="draw debug normal polygons")

    args = parser.parse_args()

    if len(args.texture) != 2:
        print("Please, provide exactly 2 values to the --texture argument")
        sys.exit(1)
    for dim in args.texture:
        if dim not in VALID_TEXTURE_SIZES:
            print(f"Invalid texture size {dim}. Valid values: {VALID_TEXTURE_SIZES}")
            sys.exit(1)

    os.makedirs(args.output, exist_ok=True)
    extension_mesh = "_dsm.bin" if args.bin else ".dsm"
    extension_anim = "_dsa.bin" if args.bin else ".dsa"

    try:
        convert(args.model, args.name, args.output, args.texture,
                extension_mesh, extension_anim, args.export_base_pose,
                args.draw_normal_polygons, not args.no_flip_winding)
    except GLTFFormatError as e:
        print("ERROR: Invalid glTF file: " + str(e))
        traceback.print_exc()
        sys.exit(1)
    except BaseException as e:
        print("ERROR: " + str(e))
        traceback.print_exc()
        sys.exit(1)

    print("Done!")
    sys.exit(0)
