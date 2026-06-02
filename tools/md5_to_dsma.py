#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
#
# Copyright (c) 2022 Antonio Niño Díaz <antonio_nd@outlook.com>

import os

from collections import namedtuple
from math import sqrt

from dsma_common import (Vector, Quaternion, Joint, joint_info_to_m4x3,
                         save_animation, emit_triangles_to_dsm)

class MD5FormatError(Exception):
    pass

VALID_TEXTURE_SIZES = [8, 16, 32, 64, 128, 256, 512, 1024]

def is_valid_texture_size(size):
    return size in VALID_TEXTURE_SIZES

def assert_num_args(cmd, real, expected, tokens):
    if real != expected:
        raise MD5FormatError(f"Unexpected nargs for '{cmd}' ({real} != {expected}): {tokens}")

def quaternion_fill_incomplete_w(v):
    """
    This expands an incomplete quaternion, not a regular vector. This is
    needed if a quaternion is stored as the components x, y and z and it is
    expected that the code will fill the value of w.
    """
    t = 1.0 - (v[0] * v[0]) - (v[1] * v[1]) - (v[2] * v[2])
    if t < 0:
        w = 0
    else:
        w = -sqrt(t)
    return Quaternion(w, v[0], v[1], v[2])

def apply_blender_fix(frames, blender_fix):
    """
    Blender uses Z as "up", the DS uses Y as "up". The bones in the DSA store
    absolute transforms, so every bone must be rotated by -90 degrees on the X
    axis to match the DS coordinate system. The DSM stores vertices in joint
    space (invariant under this global rotation) so it is left untouched.
    """
    if not blender_fix:
        return frames

    q_rot = Quaternion(0.7071068, -0.7071068, 0, 0)
    fixed = []
    for joints in frames:
        new_joints = []
        for joint in joints:
            orient = q_rot.mul(joint.orient)
            pos = Vector(joint.pos.x, joint.pos.z, -joint.pos.y)
            new_joints.append(Joint(joint.name, joint.parent, pos, orient))
        fixed.append(new_joints)
    return fixed

def parse_md5mesh(input_file):
    Vert = namedtuple("Vert", "st startWeight countWeight")
    Weight = namedtuple("Weight", "joint bias pos")
    Mesh = namedtuple("Mesh", "numverts verts numtris tris numweights weights")

    joints = []
    meshes = []

    with open(input_file, 'r') as md5mesh_file:

        numJoints = None
        numMeshes = None

        # This can have three values:
        # - "root": Parsing commands in the md5mesh outside of any group
        # - "joints": Inside a "joints" node.
        # - "mesh": Inside a "mesh" node.
        mode = "root"

        # Temporary variables used to store mesh information before packing it
        numverts = None
        verts = None
        numtris = None
        tris = None
        numweights = None
        weights = None

        for line in md5mesh_file:
            # Remove comments
            line = line.split('//')[0]

            # Parse line
            tokens = line.split()

            if len(tokens) == 0:  # Empty line
                continue

            cmd = tokens[0]
            tokens = tokens[1:]
            nargs = len(tokens)

            if mode == "root":
                if cmd == 'MD5Version':
                    assert_num_args('MD5Version', nargs, 1, tokens)
                    version = int(tokens[0])
                    if version != 10:
                        raise MD5FormatError(f"Invalid 'MD5Version': {version} != 10")

                elif cmd == 'commandline':
                    # Ignore this
                    pass

                elif cmd == 'numJoints':
                    assert_num_args('numJoints', nargs, 1, tokens)
                    numJoints = int(tokens[0])
                    if numJoints == 0:
                        raise MD5FormatError(f"'numJoints' is 0")

                elif cmd == 'numMeshes':
                    assert_num_args('numMeshes', nargs, 1, tokens)
                    numMeshes = int(tokens[0])
                    if numMeshes == 0:
                        raise MD5FormatError(f"'numMeshes' is 0")

                elif cmd == 'joints':
                    assert_num_args('joints', nargs, 1, tokens)
                    if tokens[0] != '{':
                        raise MD5FormatError(f"Unexpected token for 'joints': {tokens}")
                    if numJoints is None:
                        raise MD5FormatError("'joints' command before 'numJoints'")
                    mode = "joints"

                elif cmd == 'mesh':
                    assert_num_args('mesh', nargs, 1, tokens)
                    if tokens[0] != '{':
                        raise MD5FormatError(f"Unexpected token for 'mesh': {tokens}")
                    if numMeshes is None:
                        raise MD5FormatError("'mesh' command before 'numMeshes'")
                    mode = "mesh"

                else:
                    print(f"Ignored unsupported command: {cmd} {tokens}")

            elif mode == "joints":
                if cmd == '}':
                    if nargs > 0:
                        raise MD5FormatError(f"Unexpected tokens after 'joints {{}}': {tokens}")
                    mode = "root"
                else:
                    _, name, line = line.split('"')
                    tokens = line.strip().split(" ")
                    nargs = len(tokens)

                    assert_num_args('joint entry', nargs, 11, tokens)

                    parent = int(tokens[0])

                    if tokens[1] != '(':
                        raise MD5FormatError(f"Unexpected token 1 for joint': {tokens}")
                    pos = Vector(float(tokens[2]), float(tokens[3]), float(tokens[4]))
                    if tokens[5] != ')':
                        raise MD5FormatError(f"Unexpected token 5 for joint': {tokens}")

                    if tokens[6] != '(':
                        raise MD5FormatError(f"Unexpected token 6 for joint': {tokens}")
                    orient = (float(tokens[7]), float(tokens[8]), float(tokens[9]))
                    q_orient = quaternion_fill_incomplete_w(orient)
                    if tokens[10] != ')':
                        raise MD5FormatError(f"Unexpected token 10 for joint': {tokens}")

                    joints.append(Joint(name, parent, pos, q_orient))

            elif mode == "mesh":
                if cmd == '}':
                    if nargs != 0:
                        raise MD5FormatError(f"Unexpected tokens after 'mesh {{}}': {tokens}")
                    mode = "root"

                    meshes.append(Mesh(numverts, verts, numtris, tris, numweights, weights))

                    numverts = None
                    verts = None
                    numtris = None
                    tris = None
                    numweights = None
                    weights = None

                elif cmd == 'shader':
                    # Ignore this
                    pass

                elif cmd == 'numverts':
                    assert_num_args('numverts', nargs, 1, tokens)
                    numverts = int(tokens[0])
                    verts = [None] * numverts

                elif cmd == 'vert':
                    assert_num_args('vert', nargs, 7, tokens)
                    if numverts is None:
                        raise MD5FormatError("'vert' command before 'numverts'")

                    index = int(tokens[0])

                    if tokens[1] != '(':
                        raise MD5FormatError(f"Unexpected token 1 for vert': {tokens}")
                    st = (float(tokens[2]), float(tokens[3]))
                    if tokens[4] != ')':
                        raise MD5FormatError(f"Unexpected token 4 for vert': {tokens}")

                    startWeight = int(tokens[5])
                    countWeight = int(tokens[6])

                    if countWeight != 1:
                        raise MD5FormatError(
                            f"Vertex with {countWeight} weights detected, but this tool "
                            "only supports vertices with one weight. Ensure that all your "
                            "vertices are assigned exactly one weight with a bias of 1.0."
                        )

                    verts[index] = Vert(st, startWeight, countWeight)

                elif cmd == 'numtris':
                    assert_num_args('numtris', nargs, 1, tokens)
                    numtris = int(tokens[0])
                    tris = [None] * numtris

                elif cmd == 'tri':
                    assert_num_args('tri', nargs, 4, tokens)
                    if numtris is None:
                        raise MD5FormatError("'tri' command before 'numtris'")

                    index = int(tokens[0])
                    # Reverse order so that they face the right direction
                    vertIndices = (int(tokens[3]), int(tokens[2]), int(tokens[1]))

                    tris[index] = vertIndices

                elif cmd == 'numweights':
                    assert_num_args('numweights', nargs, 1, tokens)
                    numweights = int(tokens[0])
                    weights = [None] * numweights

                elif cmd == 'weight':
                    assert_num_args('weight', nargs, 8, tokens)
                    if numverts is None:
                        raise MD5FormatError("'weight' command before 'numweights'")

                    index = int(tokens[0])
                    jointIndex = int(tokens[1])
                    bias = float(tokens[2])

                    if bias != 1.0:
                        raise MD5FormatError(
                            f"Weight with bias {bias} detected, but this tool only"
                            "supports weights with bias equal to 1.0. Ensure that all"
                            "your vertices are assigned exactly one weight with a"
                            "bias of 1.0."
                        )

                    if tokens[3] != '(':
                        raise MD5FormatError(f"Unexpected token 3 for weight': {tokens}")
                    pos = Vector(float(tokens[4]), float(tokens[5]), float(tokens[6]))
                    if tokens[7] != ')':
                        raise MD5FormatError(f"Unexpected token 7 for weight': {tokens}")

                    weights[index] = Weight(jointIndex, bias, pos)

                else:
                    print(f"Ignored unsupported command: {cmd} {tokens}")

        if mode != "root":
            raise MD5FormatError("Unexpected end of file (expected '}')")

    realJoints = len(joints)
    if numJoints != realJoints:
        raise MD5FormatError(f"Incorrect number of joints: {numJoints} != {realJoints}")

    realMeshes = len(meshes)
    if numJoints != realJoints:
        raise MD5FormatError(f"Incorrect number of joints: {numJoints} != {realJoints}")

    return (joints, meshes)

def parse_md5anim(input_file):
    joints = []
    frames = []

    with open(input_file, 'r') as md5anim_file:

        numFrames = None
        numJoints = None

        baseframe = []
        hierarchy = []

        # This can have three values:
        # - "root": Parsing commands in the md5mesh outside of any group
        # - "hierarchy": Inside a "hierarchy" node.
        # - "bounds": Inside a "bounds" node.
        # - "baseframe": Inside a "baseframe" node.
        # - "frame": Inside a "frame" node.
        mode = "root"

        frame_index = None

        for line in md5anim_file:
            # Remove comments
            line = line.split('//')[0]

            # Parse line
            tokens = line.split()

            if len(tokens) == 0:  # Empty line
                continue

            cmd = tokens[0]
            tokens = tokens[1:]
            nargs = len(tokens)

            if mode == "root":
                if cmd == 'MD5Version':
                    assert_num_args('MD5Version', nargs, 1, tokens)
                    version = int(tokens[0])
                    if version != 10:
                        raise MD5FormatError(f"Invalid 'MD5Version': {version} != 10")

                elif cmd == 'commandline':
                    # Ignore this
                    pass

                elif cmd == 'numFrames':
                    assert_num_args('numFrames', nargs, 1, tokens)
                    numFrames = int(tokens[0])
                    if numFrames == 0:
                        raise MD5FormatError(f"'numFrames' is 0")
                    frames = [None] * numFrames

                elif cmd == 'numJoints':
                    assert_num_args('numJoints', nargs, 1, tokens)
                    numJoints = int(tokens[0])
                    if numJoints == 0:
                        raise MD5FormatError(f"'numJoints' is 0")

                elif cmd == 'frameRate':
                    # Ignore this
                    pass

                elif cmd == 'numAnimatedComponents':
                    # Ignore this
                    pass

                elif cmd == 'hierarchy':
                    assert_num_args('hierarchy', nargs, 1, tokens)
                    if tokens[0] != '{':
                        raise MD5FormatError(f"Unexpected token for 'hierarchy': {tokens}")
                    mode = "hierarchy"

                elif cmd == 'bounds':
                    assert_num_args('bounds', nargs, 1, tokens)
                    if tokens[0] != '{':
                        raise MD5FormatError(f"Unexpected token for 'bounds': {tokens}")
                    mode = "bounds"

                elif cmd == 'baseframe':
                    assert_num_args('baseframe', nargs, 1, tokens)
                    if tokens[0] != '{':
                        raise MD5FormatError(f"Unexpected token for 'baseframe': {tokens}")
                    mode = "baseframe"

                elif cmd == 'frame':
                    assert_num_args('frame', nargs, 2, tokens)
                    frame_index = int(tokens[0])

                    if tokens[1] != '{':
                        raise MD5FormatError(f"Unexpected token for 'frame': {tokens}")
                    if numFrames is None:
                        raise MD5FormatError("'frame' command before 'numFrames'")
                    mode = "frame"
                    joints = []

                else:
                    print(f"Ignored unsupported command: {cmd} {tokens}")

            elif mode == "hierarchy":
                if cmd == '}':
                    if nargs > 0:
                        raise MD5FormatError(f"Unexpected tokens after 'hierarchy {{}}': {tokens}")
                    mode = "root"
                else:
                    _, name, line = line.split('"')
                    tokens = line.strip().split(" ")
                    nargs = len(tokens)

                    assert_num_args('hierarchy entry', nargs, 3, tokens)

                    parent_index = int(tokens[0])
                    flags = int(tokens[1])
                    if flags != 63:
                        raise MD5FormatError(f"Unexpected flags in hierarchy: {flags}")
                    frame_data_index = int(tokens[2])

                    hierarchy.append(parent_index)

            elif mode == "bounds":
                if cmd == '}':
                    if nargs > 0:
                        raise MD5FormatError(f"Unexpected tokens after 'bounds {{}}': {tokens}")
                    mode = "root"
                else:
                    # Ignore everything else
                    pass

            elif mode == "baseframe":
                if cmd == '}':
                    if nargs > 0:
                        raise MD5FormatError(f"Unexpected tokens after 'baseframe {{}}': {tokens}")
                    mode = "root"
                else:
                    values = line.strip().split()
                    assert_num_args('baseframe joint', len(values), 10, values)

                    if values[0] != '(':
                        raise MD5FormatError(f"Unexpected token 0 for baseframe': {values}")
                    pos = Vector(float(values[1]), float(values[2]), float(values[3]))
                    if values[4] != ')':
                        raise MD5FormatError(f"Unexpected token 4 for baseframe': {values}")

                    if values[5] != '(':
                        raise MD5FormatError(f"Unexpected token 5 for baseframe': {values}")
                    orient = (float(values[6]), float(values[7]), float(values[8]))
                    q_orient = quaternion_fill_incomplete_w(orient)
                    if values[9] != ')':
                        raise MD5FormatError(f"Unexpected token 9 for baseframe': {values}")

                    baseframe.append(Joint("", -1, pos, q_orient))

            elif mode == "frame":
                if cmd == '}':
                    if nargs > 0:
                        raise MD5FormatError(f"Unexpected tokens after 'frame {{}}': {tokens}")
                    mode = "root"

                    # Now that the frame has been read, process the real
                    # positions and orientations of the bones before storing
                    # them.

                    transformed_joints = []

                    for joint, parent_index in zip(joints, hierarchy):
                        if parent_index == -1:
                            # Root bone
                            transformed_joints.append(joint)
                        else:
                            parent_pos = transformed_joints[parent_index].pos
                            parent_orient = transformed_joints[parent_index].orient

                            this_pos = joint.pos
                            this_orient = joint.orient

                            q = parent_orient
                            qt = q.complement()
                            q_pos_delta = q.mul(this_pos.to_q()).mul(qt)
                            pos_delta = q_pos_delta.to_v3()

                            pos = parent_pos.add(pos_delta)
                            orient = parent_orient.mul(this_orient).normalize()

                            transformed_joints.append(Joint("", -1, pos, orient))

                    frames[frame_index] = transformed_joints
                else:
                    values = line.strip().split()
                    assert_num_args('frame joint', len(values), 6, values)

                    pos = Vector(float(values[0]), float(values[1]), float(values[2]))

                    orient = (float(values[3]), float(values[4]), float(values[5]))
                    q_orient = quaternion_fill_incomplete_w(orient)

                    joints.append(Joint("", -1, pos, q_orient))

        if mode != "root":
            raise MD5FormatError("Unexpected end of file (expected '}')")

    realJoints = len(joints)
    if numJoints != realJoints:
        raise MD5FormatError(f"Incorrect number of joints: {numJoints} != {realJoints}")

    realFrames = len(frames)
    if numFrames != realFrames:
        raise MD5FormatError(f"Incorrect number of frames: {numFrames} != {realFrames}")

    return frames

def convert_md5mesh(model_file, name, output_folder, texture_size,
                    draw_normal_polygons, extension_mesh, extension_anim,
                    blender_fix, export_base_pose):

    print(f"Converting model: {model_file}")

    # Parse md5mesh file
    joints, meshes = parse_md5mesh(model_file)

    print(f"Loaded {len(joints)} joint(s) and {len(meshes)} mesh(es).")

    if len(meshes) > 1:
        print("WARNING: More than one mesh found. All meshes will share the same "
              "texture. If you want them to have different textures, you must use "
              "multiple .md5mesh files.")

    if export_base_pose:
        print("Converting base pose...")

        save_animation(apply_blender_fix([joints], blender_fix),
                       os.path.join(output_folder, f"{name}{extension_anim}"))

    print("Converting meshes...")

    # Flatten all meshes into a single neutral triangle list. Each vertex becomes
    # (joint_index, joint-space position, st) — the form the shared emit expects.
    triangles = []
    for mesh in meshes:
        print(f"  Vertices: {mesh.numverts}")
        print(f"  Tris:     {mesh.numtris}")
        print(f"  Weights:  {mesh.numweights}")
        for tri in mesh.tris:
            triangle = []
            for i in tri:
                vert = mesh.verts[i]
                weight = mesh.weights[vert.startWeight]
                triangle.append((weight.joint, weight.pos, vert.st))
            triangles.append(triangle)

    print("  Generating display list...")
    emit_triangles_to_dsm(joints, triangles, texture_size,
                          os.path.join(output_folder, f"{name}{extension_mesh}"),
                          draw_normal_polygons)


def convert_md5anim(name, output_folder, anim_file, skip_frames, extension_anim,
                    blender_fix):

    print(f"Converting animation: {anim_file}")

    frames = parse_md5anim(anim_file)

    # Create name of animation based on file name
    file_basename = os.path.basename(anim_file).replace(".md5anim", "")
    anim_name = file_basename.replace(".", "_").lower()

    frames = frames[::skip_frames+1]
    save_animation(apply_blender_fix(frames, blender_fix),
                   os.path.join(output_folder, f"{name}_{anim_name}{extension_anim}"))


if __name__ == "__main__":

    import argparse
    import sys
    import traceback

    print("md5_to_dsma v0.1.1")
    print("Copyright (c) 2022-2024 Antonio Niño Díaz <antonio_nd@outlook.com>")
    print("All rights reserved")
    print("")

    parser = argparse.ArgumentParser(
            description='Converts md5mesh and md5anim files into DSM and DSA files.')

    # Required arguments
    parser.add_argument("--name", required=True,
                        help="model name to be used in output files")
    parser.add_argument("--output", required=True,
                        help="output folder")

    # Optional arguments
    parser.add_argument("--model", required=False, type=str, default=None,
                        help="input md5mesh file")
    parser.add_argument("--texture", required=False, type=int, default=[],
                        nargs="+", action="extend",
                        help="texture width and height (e.g. '--texture 32 64')")
    parser.add_argument("--anims", required=False, type=str, default=[],
                        nargs="+", action="extend",
                        help="list of md5anim files to convert")
    parser.add_argument("--bin", required=False,
                        action='store_true',
                        help="add '.bin' to the name of the output files")
    parser.add_argument("--blender-fix", required=False,
                        action='store_true',
                        help="rotate model -90 degrees on X axis to match Blender's orientation")
    parser.add_argument("--export-base-pose", required=False,
                        action='store_true',
                        help="export base pose of a md5mesh as a DSA file")
    parser.add_argument("--skip-frames", required=False,
                        default=0, type=int,
                        help="number of frames to skip in an animation (0 = export all, 1 = export half, 2 = export 33%%, etc)")
    parser.add_argument("--draw-normal-polygons", required=False,
                        action='store_true',
                        help="draw polygons with the shape of normals for debugging")

    args = parser.parse_args()

    if args.model is not None:
        if len(args.texture) != 2:
            print("Please, provide exactly 2 values to the --texture argument")
            sys.exit(1)

        if not is_valid_texture_size(args.texture[0]):
            print(f"Invalid texture width. Valid values: {VALID_TEXTURE_SIZES}")
            sys.exit(1)

        if not is_valid_texture_size(args.texture[1]):
            print(f"Invalid texture height. Valid values: {VALID_TEXTURE_SIZES}")
            sys.exit(1)

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    # Add '.bin' to the name of the files if requested
    extension_mesh = "_dsm.bin" if args.bin else ".dsm"
    extension_anim = "_dsa.bin" if args.bin else ".dsa"

    try:
        if args.model is not None:
            convert_md5mesh(args.model, args.name, args.output, args.texture,
                            args.draw_normal_polygons, extension_mesh,
                            extension_anim, args.blender_fix,
                            args.export_base_pose)

        for anim_file in args.anims:
            convert_md5anim(args.name, args.output, anim_file, args.skip_frames,
                            extension_anim, args.blender_fix)

    except MD5FormatError as e:
        print("ERROR: Invalid MD5 file: " + str(e))
        traceback.print_exc()
        sys.exit(1)
    except BaseException as e:
        print("ERROR: " + str(e))
        traceback.print_exc()
        sys.exit(1)

    print("Done!")

    sys.exit(0)
