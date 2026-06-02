#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
#
# Copyright (c) 2022 Antonio Niño Díaz <antonio_nd@outlook.com>
# Copyright (c) 2026 Affinity contributors
#
# Shared backend for the DSMA converters. Both md5_to_dsma and gltf_to_dsma
# produce the same neutral intermediate (absolute bind-pose joints + a flat list
# of triangles whose vertices are expressed in joint space) and hand it to the
# emit functions here. This keeps the DSM display-list generation and the DSA
# fixed-point layout in exactly one place regardless of the source format.

from collections import namedtuple
from math import sqrt

from display_list import DisplayList, float_to_f32

# A joint of the (absolute) bind pose / animation frame. pos is a Vector, orient
# is a Quaternion. parent/name are informational and may be unused by emit.
Joint = namedtuple("Joint", "name parent pos orient")


class Quaternion():
    def __init__(self, w, x, y, z):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def to_v3(self):
        return Vector(self.x, self.y, self.z)

    def complement(self):
        return Quaternion(self.w, -self.x, -self.y, -self.z)

    def normalize(self):
        mag = sqrt((self.w ** 2) + (self.x ** 2) + (self.y ** 2) + (self.z ** 2))
        return Quaternion(self.w / mag, self.x / mag, self.y / mag, self.z / mag)

    def mul(self, other):
        w = (self.w * other.w) - (self.x * other.x) - (self.y * other.y) - (self.z * other.z)
        x = (self.x * other.w) + (self.w * other.x) + (self.y * other.z) - (self.z * other.y)
        y = (self.y * other.w) + (self.w * other.y) + (self.z * other.x) - (self.x * other.z)
        z = (self.z * other.w) + (self.w * other.z) + (self.x * other.y) - (self.y * other.x)
        return Quaternion(w, x, y, z)


class Vector():
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

    def to_q(self):
        return Quaternion(0, self.x, self.y, self.z)

    def length(self):
        return sqrt((self.x ** 2) + (self.y ** 2) + (self.z ** 2))

    def normalize(self):
        mag = self.length()
        return Vector(self.x / mag, self.y / mag, self.z / mag)

    def add(self, other):
        return Vector(self.x + other.x, self.y + other.y, self.z + other.z)

    def sub(self, other):
        return Vector(self.x - other.x, self.y - other.y, self.z - other.z)

    def cross(self, other):
        x = (self.y * other.z) - (other.y * self.z)
        y = (self.z * other.x) - (other.z * self.x)
        z = (self.x * other.y) - (other.x * self.y)
        return Vector(x, y, z)

    def mul_m4x3(self, m):
        x = (self.x * m[0][0]) + (self.y * m[0][1]) + (self.z * m[0][2]) + (m[0][3] * 1)
        y = (self.x * m[1][0]) + (self.y * m[1][1]) + (self.z * m[1][2]) + (m[1][3] * 1)
        z = (self.x * m[2][0]) + (self.y * m[2][1]) + (self.z * m[2][2]) + (m[2][3] * 1)
        return Vector(x, y, z)


def joint_info_to_m4x3(q, trans):
    """
    Generate a 4x3 matrix that represents a rotation and a translation.
    q is a Quaternion with an orientation, trans is a Vector with a translation.
    """
    wx = 2 * q.w * q.x
    wy = 2 * q.w * q.y
    wz = 2 * q.w * q.z
    x2 = 2 * q.x * q.x
    xy = 2 * q.x * q.y
    xz = 2 * q.x * q.z
    y2 = 2 * q.y * q.y
    yz = 2 * q.y * q.z
    z2 = 2 * q.z * q.z

    return [[1 - y2 - z2,     xy - wz,     xz + wy, trans.x],
            [    xy + wz, 1 - x2 - z2,     yz - wx, trans.y],
            [    xz - wy,     yz + wx, 1 - x2 - y2, trans.z]]


def save_animation(frames, output_file):
    """
    Write a DSA file. frames is a list of frames; each frame is a list of Joint
    holding ABSOLUTE (already hierarchy-composed) pos+orient for every bone. The
    caller is responsible for any coordinate fix-ups before calling this.
    """
    version = 1
    num_frames = len(frames)
    num_bones = len(frames[0])

    u32_array = [version, num_frames, num_bones]

    for joints in frames:
        if num_bones != len(joints):
            raise ValueError("Different number of bones across frames")

        for joint in joints:
            pos = [float_to_f32(joint.pos.x), float_to_f32(joint.pos.y),
                   float_to_f32(joint.pos.z)]
            orient = [float_to_f32(joint.orient.w), float_to_f32(joint.orient.x),
                      float_to_f32(joint.orient.y), float_to_f32(joint.orient.z)]
            u32_array.extend(pos)
            u32_array.extend(orient)

    with open(output_file, "wb") as f:
        for u32 in u32_array:
            b = [u32 & 0xFF, (u32 >> 8) & 0xFF, (u32 >> 16) & 0xFF, (u32 >> 24) & 0xFF]
            f.write(bytearray(b))


def emit_triangles_to_dsm(joints, triangles, texture_size, output_file,
                          draw_normal_polygons=False):
    """
    Build a DSM display list from a neutral triangle list and save it.

    joints     : list of Joint (absolute bind pose) — used to turn joint-space
                 vertices into world space (for face normals) and to rotate the
                 face normal into each joint's local space.
    triangles  : list of triangles; each triangle is a 3-tuple/list of vertices,
                 each vertex a tuple (joint_index, pos, st) where
                   joint_index : int        bone this vertex is rigidly bound to
                   pos         : Vector      vertex position in that joint's space
                   st          : (u, v)      texcoords in 0..1 range
    texture_size : (w, h) used to scale the 0..1 st into texel coordinates.

    The geometry/normal math is identical to the original md5_to_dsma emit so
    that both front-ends produce byte-comparable output for equivalent input.
    """
    dl = DisplayList()
    dl.switch_vtxs("triangles")

    # Each joint matrix lives in one slot of the DS matrix stack. They are packed
    # at the top of the stack so that index 0 maps to (31 - num_joints).
    base_matrix = 30 - len(joints) + 1
    last_joint_index = None

    # Per-triangle face normals, computed in world space from the bind pose.
    tri_normal = []
    for tri in triangles:
        world = []
        for (joint_index, pos, _st) in tri:
            joint = joints[joint_index]
            m = joint_info_to_m4x3(joint.orient, joint.pos)
            world.append(pos.mul_m4x3(m))

        a = world[0].sub(world[1])
        b = world[1].sub(world[2])
        n = a.cross(b)
        if n.length() > 0:
            tri_normal.append(n.normalize())
        else:
            tri_normal.append(Vector(0, 0, 0))

    for tri, norm in zip(triangles, tri_normal):
        finals = []
        for (joint_index, pos, st) in tri:
            # Texture (DS expects (0,0) at top-left, same as glTF and MD5).
            dl.texcoord(st[0] * texture_size[0], st[1] * texture_size[1])

            # Load this vertex's joint matrix. When drawing debug normal polygons
            # it has to be reloaded every time because the normal polygon restores
            # a different matrix.
            if draw_normal_polygons or joint_index != last_joint_index:
                dl.mtx_restore(base_matrix + joint_index)
                last_joint_index = joint_index

            joint = joints[joint_index]

            # Rotate the world-space face normal into the joint's local space, so
            # that the runtime bone matrix rotates it back to world space.
            q = joint.orient
            qt = q.complement()
            n = qt.mul(norm.to_q()).mul(q).to_v3()
            if n.length() > 0:
                n = n.normalize()
            dl.normal(n.x, n.y, n.z)

            # The vertex is already expressed in joint space.
            dl.vtx(pos.x, pos.y, pos.z)

            if draw_normal_polygons:
                v = pos.to_q()
                delta = q.mul(v).mul(qt).to_v3()
                finals.append(joint.pos.add(delta))

        if draw_normal_polygons:
            dl.mtx_restore(1)
            vert_avg = Vector((finals[0].x + finals[1].x + finals[2].x) / 3,
                              (finals[0].y + finals[1].y + finals[2].y) / 3,
                              (finals[0].z + finals[1].z + finals[2].z) / 3)
            vert_avg_end = vert_avg.add(norm)
            dl.texcoord(0, 0)

            dl.color(1, 0, 0)
            dl.vtx(vert_avg.x + 0.1, vert_avg.y, vert_avg.z)
            dl.vtx(vert_avg.x, vert_avg.y, vert_avg.z)
            dl.color(0, 1, 0)
            dl.vtx(vert_avg_end.x, vert_avg_end.y, vert_avg_end.z)

            dl.color(1, 0, 0)
            dl.vtx(vert_avg.x, vert_avg.y, vert_avg.z)
            dl.vtx(vert_avg.x, vert_avg.y + 0.1, vert_avg.z)
            dl.color(0, 1, 0)
            dl.vtx(vert_avg_end.x, vert_avg_end.y, vert_avg_end.z)

            dl.color(1, 0, 0)
            dl.vtx(vert_avg.x, vert_avg.y, vert_avg.z)
            dl.vtx(vert_avg.x, vert_avg.y, vert_avg.z + 0.1)
            dl.color(0, 1, 0)
            dl.vtx(vert_avg_end.x, vert_avg_end.y, vert_avg_end.z)

    dl.end_vtxs()
    dl.finalize()
    dl.save_to_file(output_file)
