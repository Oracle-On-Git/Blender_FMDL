# SPDX-License-Identifier: GPL-3.0-or-later
"""
Pure-Python (numpy only) reader for Fox Engine .fmdl files (FMDL 2.03).

This module has no dependency on Blender so it can be tested on its own.

File layout (as decoded from real files, cross-checked against the MGSV
modding wiki "Fmdl" page):

    0x00  "FMDL", float version
    0x20  u32 section0 entry count, u32 section1 entry count
    0x28  u32 section0 offset, length, section1 offset, length
    0x40  section0 table: N x (u16 type, u16 count, u32 offset)
          section1 table: M x (u32 type, u32 offset, u32 length)

Section 0 holds fixed-size records ("features"), section 1 holds raw blobs
(material vectors, the vertex/index buffer block, and the string table).
"""

import struct
from dataclasses import dataclass, field

import numpy as np


class FmdlError(Exception):
    pass


# --- Section 0 feature types -------------------------------------------------
F_BONE = 0x00
F_MESH_GROUP = 0x01
F_MESH_GROUP_ENTRY = 0x02
F_MESH = 0x03
F_MATERIAL_INSTANCE = 0x04
F_BONE_GROUP = 0x05
F_TEXTURE = 0x06
F_MATERIAL_PARAM = 0x07
F_MATERIAL = 0x08
F_MESH_FORMAT = 0x09
F_BUFFER_OFFSET = 0x0A
F_VERTEX_FORMAT = 0x0B
F_STRING = 0x0C
F_BOUNDING_BOX = 0x0D
F_BUFFER = 0x0E
F_LOD_INFO = 0x10
F_FACE_INFO = 0x11

# Known record sizes (bytes). Others are derived from the table gaps.
KNOWN_SIZES = {
    F_BONE: 48, F_MESH_GROUP: 8, F_MESH_GROUP_ENTRY: 32, F_MESH: 48,
    F_MATERIAL_INSTANCE: 16, F_TEXTURE: 4, F_MATERIAL_PARAM: 4,
    F_MATERIAL: 4, F_MESH_FORMAT: 8, F_BUFFER_OFFSET: 8, F_VERTEX_FORMAT: 4,
    F_STRING: 8, F_BOUNDING_BOX: 32, F_BUFFER: 16, F_LOD_INFO: 16,
    F_FACE_INFO: 8,
}

# --- Section 1 blob types ----------------------------------------------------
S1_MATERIAL_VECTORS = 0
S1_MESH_DATA = 2
S1_STRINGS = 3

# --- Vertex usages / data formats -------------------------------------------
U_POSITION = 0
U_WEIGHTS = 1
U_NORMAL = 2
U_COLOR = 3
U_BONE_INDICES = 7
U_UV0, U_UV1, U_UV2, U_UV3 = 8, 9, 10, 11
U_TANGENT = 14

# format id -> (numpy dtype, component count, normalized)
VERTEX_FORMATS = {
    1: ("<f4", 3, False),
    2: ("<f4", 2, False),
    3: ("<f4", 4, False),
    4: ("u1", 4, True),
    5: ("u1", 4, False),
    6: ("<f2", 4, False),
    7: ("<f2", 2, False),
    8: ("u1", 4, True),    # unorm8 x4: bone weights, colors (seen in real files)
    9: ("u1", 4, False),   # uint8 x4: bone indices (seen in real files)
}


@dataclass
class Bone:
    name: str
    parent: int
    world_pos: tuple
    local_pos: tuple


@dataclass
class MaterialInstance:
    name: str
    material_name: str = ""
    material_type: str = ""
    textures: list = field(default_factory=list)  # [(param_name, file, path)]
    vectors: list = field(default_factory=list)   # [(param_name, (x,y,z,w))]


@dataclass
class Mesh:
    index: int
    alpha: int
    shadow: int
    flags: int
    material_instance: int
    bone_group: int
    mesh_format: int
    vertex_count: int
    face_base: int
    face_index_count: int
    first_face_info: int
    lod_count: int = 1
    group: int = -1


@dataclass
class MeshData:
    positions: np.ndarray
    faces: np.ndarray                # (M, 3) int32
    normals: np.ndarray = None
    tangents: np.ndarray = None
    uvs: list = field(default_factory=list)      # list of (N, 2) float32
    colors: np.ndarray = None                    # (N, 4) float32
    weights: np.ndarray = None                   # (N, 4) float32
    bone_indices: np.ndarray = None              # (N, 4) int32
    elements: list = field(default_factory=list)  # [(usage, fmt, buffer, offset, stride)]
    warnings: list = field(default_factory=list)


class FmdlModel:
    def __init__(self, data: bytes):
        self.data = data
        self.warnings = []
        self._parse_header()
        self._parse_strings()
        self._parse_tables()
        self._parse_materials()

    # -- low level ------------------------------------------------------------
    def _parse_header(self):
        d = self.data
        if len(d) < 0x40 or d[:4] != b"FMDL":
            raise FmdlError("Not an FMDL file (bad magic).")
        self.version = struct.unpack_from("<f", d, 4)[0]
        if abs(self.version - 2.03) > 0.001:
            self.warnings.append(
                f"FMDL version {self.version:.2f} (only 2.03 was tested)."
            )
        n0, n1 = struct.unpack_from("<II", d, 0x20)
        self.s0_off, self.s0_len, self.s1_off, self.s1_len = struct.unpack_from(
            "<IIII", d, 0x28
        )
        if self.s1_off + self.s1_len > len(d):
            raise FmdlError("Section 1 extends past end of file (truncated?).")

        # section 0 feature table
        raw = [struct.unpack_from("<HHI", d, 0x40 + i * 8) for i in range(n0)]
        self.features = {}  # type -> (count, abs_offset, record_size)
        by_off = sorted(raw, key=lambda e: e[2])
        for t, c, o in raw:
            nxt = [e[2] for e in by_off if e[2] > o]
            end = nxt[0] if nxt else self.s0_len
            size = KNOWN_SIZES.get(t) or ((end - o) // c if c else 0)
            self.features[t] = (c, self.s0_off + o, size)

        # section 1 blob table
        base = 0x40 + n0 * 8
        self.blobs = {}
        for i in range(n1):
            t, o, l = struct.unpack_from("<III", d, base + i * 12)
            self.blobs[t] = (self.s1_off + o, l)

    def rec(self, ftype, index):
        """Raw bytes of record `index` of feature `ftype`."""
        c, off, size = self.features[ftype]
        if not 0 <= index < c:
            raise FmdlError(f"Record {index} out of range for feature 0x{ftype:X}")
        return self.data[off + index * size: off + (index + 1) * size]

    def count(self, ftype):
        return self.features.get(ftype, (0, 0, 0))[0]

    def has(self, ftype):
        return self.count(ftype) > 0

    def _parse_strings(self):
        self._strings = []
        if F_STRING not in self.features or S1_STRINGS not in self.blobs:
            return
        sbase, _ = self.blobs[S1_STRINGS]
        for i in range(self.count(F_STRING)):
            _sec, length, off = struct.unpack_from("<HHI", self.rec(F_STRING, i))
            raw = self.data[sbase + off: sbase + off + length]
            self._strings.append(raw.decode("utf-8", "replace"))

    def string(self, i):
        return self._strings[i] if 0 <= i < len(self._strings) else ""

    # -- tables ---------------------------------------------------------------
    def _parse_tables(self):
        # mesh groups
        self.mesh_groups = []  # (name, parent, invisible)
        for i in range(self.count(F_MESH_GROUP)):
            name, invis, parent, _ = struct.unpack("<HHHH", self.rec(F_MESH_GROUP, i))
            self.mesh_groups.append(
                (self.string(name), None if parent == 0xFFFF else parent, invis)
            )

        # meshes
        self.meshes = []
        for i in range(self.count(F_MESH)):
            r = self.rec(F_MESH, i)
            alpha, shadow, flags, mat, bgroup, mfmt, vcount = struct.unpack_from(
                "<BBHHHHH", r, 0
            )
            fbase, fcount, finfo = struct.unpack_from("<III", r, 16)
            self.meshes.append(
                Mesh(i, alpha, shadow, flags, mat, bgroup, mfmt, vcount,
                     fbase, fcount, finfo)
            )
        for i, m in enumerate(self.meshes):
            nxt = self.meshes[i + 1].first_face_info if i + 1 < len(self.meshes) \
                else self.count(F_FACE_INFO)
            m.lod_count = max(1, nxt - m.first_face_info)

        # mesh group entries -> which group each mesh belongs to
        for i in range(self.count(F_MESH_GROUP_ENTRY)):
            r = self.rec(F_MESH_GROUP_ENTRY, i)
            group, mcount, first = struct.unpack_from("<HHH", r, 4)
            for k in range(first, min(first + mcount, len(self.meshes))):
                self.meshes[k].group = group

        # skeleton
        self.bones = []
        for i in range(self.count(F_BONE)):
            r = self.rec(F_BONE, i)
            name, parent, _bbox, _unk = struct.unpack_from("<HHHH", r, 0)
            size = len(r)
            lo, wo = (0x10, 0x20) if size >= 48 else (size - 32, size - 16)
            local = struct.unpack_from("<4f", r, lo)
            world = struct.unpack_from("<4f", r, wo)
            self.bones.append(
                Bone(self.string(name), None if parent == 0xFFFF else parent,
                     world[:3], local[:3])
            )
        self.bone_groups = []
        for i in range(self.count(F_BONE_GROUP)):
            r = self.rec(F_BONE_GROUP, i)
            _unk, n = struct.unpack_from("<HH", r, 0)
            self.bone_groups.append(list(struct.unpack_from(f"<{n}H", r, 4)))

    def _parse_materials(self):
        vectors = []
        if S1_MATERIAL_VECTORS in self.blobs:
            off, ln = self.blobs[S1_MATERIAL_VECTORS]
            vectors = [tuple(v) for v in
                       np.frombuffer(self.data, "<f4", ln // 4, off).reshape(-1, 4)]
        textures = []
        for i in range(self.count(F_TEXTURE)):
            fn, pth = struct.unpack("<HH", self.rec(F_TEXTURE, i))
            textures.append((self.string(fn), self.string(pth)))
        params = [struct.unpack("<HH", self.rec(F_MATERIAL_PARAM, i))
                  for i in range(self.count(F_MATERIAL_PARAM))]
        mats = [struct.unpack("<HH", self.rec(F_MATERIAL, i))
                for i in range(self.count(F_MATERIAL))]

        self.material_instances = []
        for i in range(self.count(F_MATERIAL_INSTANCE)):
            r = self.rec(F_MATERIAL_INSTANCE, i)
            name, _unk, mat_id, ntex, nvec, first = struct.unpack_from("<HHHBBH", r, 0)
            mi = MaterialInstance(self.string(name))
            if mat_id < len(mats):
                mi.material_name = self.string(mats[mat_id][0])
                mi.material_type = self.string(mats[mat_id][1])
            for p in range(first, min(first + ntex + nvec, len(params))):
                pname, ref = params[p]
                if p - first < ntex:
                    if ref < len(textures):
                        mi.textures.append((self.string(pname),) + textures[ref])
                elif ref < len(vectors):
                    mi.vectors.append((self.string(pname), vectors[ref]))
            self.material_instances.append(mi)

    # -- mesh data ------------------------------------------------------------
    def lod_ranges(self, mesh):
        """List of unique (first_index, count) per LOD, LOD0 first."""
        out = []
        for k in range(mesh.lod_count):
            fi = mesh.first_face_info + k
            if fi >= self.count(F_FACE_INFO):
                break
            first, count = struct.unpack("<II", self.rec(F_FACE_INFO, fi))
            if count and (first, count) not in out:
                out.append((first, count))
        return out or [(0, mesh.face_index_count)]

    def read_mesh(self, mesh, lod=0, want_lod_faces=None):
        """Decode vertices of `mesh` plus faces of the given LOD."""
        if S1_MESH_DATA not in self.blobs:
            raise FmdlError("File has no mesh data block.")
        base, blen = self.blobs[S1_MESH_DATA]
        buf = self.data
        warnings = []

        buffers = [struct.unpack("<IIII", self.rec(F_BUFFER, i))
                   for i in range(self.count(F_BUFFER))]  # (kind, len, off, _)
        index_buf = next((b for b in buffers if b[0] == 1), None)

        mf = self.rec(F_MESH_FORMAT, mesh.mesh_format)
        n_bo, n_vf, _a, _b, first_bo, first_vf = struct.unpack_from("<BBBBHH", mf, 0)
        vfmts = [struct.unpack("<BBH", self.rec(F_VERTEX_FORMAT, first_vf + i))
                 for i in range(n_vf)]

        vc = mesh.vertex_count
        elems = {}
        seen_elements = []
        for g in range(n_bo):
            bidx, fcount, stride, ffirst, offset = struct.unpack(
                "<BBBBI", self.rec(F_BUFFER_OFFSET, first_bo + g))
            if bidx >= len(buffers):
                warnings.append(f"buffer {bidx} missing")
                continue
            gbase = base + buffers[bidx][2] + offset
            for usage, fmt, eoff in vfmts[ffirst:ffirst + fcount]:
                seen_elements.append((usage, fmt, bidx, eoff, stride))
                spec = VERTEX_FORMATS.get(fmt)
                if spec is None:
                    warnings.append(f"unsupported vertex format {fmt} (usage {usage})")
                    continue
                dt, n, norm = spec
                isz = np.dtype(dt).itemsize * n
                if eoff + isz > stride or vc == 0:
                    warnings.append(f"vertex element usage {usage} overruns stride")
                    continue
                end = gbase + eoff + (vc - 1) * stride + isz
                if end > base + blen:
                    warnings.append(f"vertex element usage {usage} out of bounds")
                    continue
                arr = np.ndarray((vc, n), dtype=dt, buffer=buf,
                                 offset=gbase + eoff,
                                 strides=(stride, np.dtype(dt).itemsize))
                arr = arr.astype(np.float32) / 255.0 if norm else arr.astype(
                    np.float32 if dt != "u1" else np.int32)
                elems.setdefault(usage, arr)

        if U_POSITION not in elems:
            raise FmdlError(f"Mesh {mesh.index} has no position data.")

        # faces: triangle lists, indices relative to the vertex block
        faces = np.zeros((0, 3), np.int32)
        if index_buf is not None:
            ranges = self.lod_ranges(mesh)
            first, count = ranges[min(lod, len(ranges) - 1)] if want_lod_faces is None \
                else want_lod_faces
            start = base + index_buf[2] + (mesh.face_base + first) * 2
            if count % 3:
                warnings.append(f"index count {count} is not a multiple of 3; "
                                "triangle strips are not supported")
                count -= count % 3
            if start + count * 2 > base + blen:
                raise FmdlError(f"Mesh {mesh.index}: index data out of bounds.")
            idx = np.frombuffer(buf, "<u2", count, start).astype(np.int32)
            faces = idx.reshape(-1, 3)
            ok = (faces < vc).all(1) & (faces[:, 0] != faces[:, 1]) \
                & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])
            if not ok.all():
                warnings.append(f"dropped {int((~ok).sum())} degenerate/invalid triangles")
                faces = faces[ok]

        md = MeshData(elems[U_POSITION][:, :3].copy(), faces, warnings=warnings,
                      elements=seen_elements)
        if U_NORMAL in elems:
            md.normals = elems[U_NORMAL][:, :3].copy()
        if U_TANGENT in elems:
            md.tangents = elems[U_TANGENT][:, :3].copy()
        for u in (U_UV0, U_UV1, U_UV2, U_UV3):
            if u in elems:
                md.uvs.append(elems[u][:, :2].copy())
        if U_COLOR in elems:
            md.colors = elems[U_COLOR].astype(np.float32)
        if U_WEIGHTS in elems:
            md.weights = elems[U_WEIGHTS].astype(np.float32)
        if U_BONE_INDICES in elems:
            md.bone_indices = elems[U_BONE_INDICES].astype(np.int32)
        return md


def load(path):
    with open(path, "rb") as f:
        return FmdlModel(f.read())


def describe(model):
    """Human-readable structural dump, useful for bug reports."""
    L = [f"FMDL v{model.version:.2f}  meshes={len(model.meshes)} bones={len(model.bones)} "
         f"bone_groups={len(model.bone_groups)} materials={len(model.material_instances)}"]
    L.append("features: " + ", ".join(
        f"0x{t:02X}x{c}({s}B)" for t, (c, _o, s) in sorted(model.features.items())))
    L.append("blobs: " + ", ".join(f"{t}:{l}B" for t, (_o, l) in sorted(model.blobs.items())))
    L.append("buffers: " + str([struct.unpack("<IIII", model.rec(F_BUFFER, i))
                                for i in range(model.count(F_BUFFER))]))
    for i, g in enumerate(model.bone_groups[:8]):
        L.append(f"bone_group[{i}] n={len(g)} {g[:12]}")
    if model.has(F_BONE_GROUP):
        L.append("bone_group[0] raw: " + model.rec(F_BONE_GROUP, 0)[:24].hex(" "))
    for m in model.meshes:
        try:
            md = model.read_mesh(m)
            el = ",".join(f"u{u}/f{f}@b{b}+{o}" for u, f, b, o, _s in md.elements)
            sk = "weights" if md.weights is not None else "-"
            sk += "+idx" if md.bone_indices is not None else ""
            mx = int(md.bone_indices.max()) if md.bone_indices is not None else -1
        except FmdlError as e:
            el, sk, mx = f"ERR {e}", "", -1
        L.append(f"mesh {m.index}: bg={m.bone_group} raw={model.rec(F_MESH, m.index)[:12].hex(' ')} "
                 f"{sk} maxidx={mx} elems={el}")
    return "\n".join(L)


if __name__ == "__main__":
    import sys
    print(describe(load(sys.argv[1])))
