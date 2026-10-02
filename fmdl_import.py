# SPDX-License-Identifier: GPL-3.0-or-later
"""Blender-side scene construction for the FMDL importer."""

import os

import bpy
import numpy as np
from mathutils import Vector

from . import fmdl_parser as P


class ImportOptions:
    def __init__(self, **kw):
        self.global_scale = 1.0
        self.mirror = True              # Fox is left-handed Y-up
        self.flip_v = True
        self.custom_normals = True
        self.import_lods = False
        self.import_armature = True
        self.load_textures = True
        self.texture_dir = ""
        self.__dict__.update(kw)


def _convert(pos, opts):
    """Fox (Y-up, left-handed) -> Blender (Z-up, right-handed)."""
    pos = np.asarray(pos, dtype=np.float32)
    if opts.mirror:
        return pos[..., [0, 2, 1]]                      # (x, z, y)
    out = pos[..., [0, 2, 1]].copy()                    # rotate: (x, -z, y)
    out[..., 1] *= -1.0
    return out


# ------------------------------------------------------------------ textures
def _build_texture_index(opts, base_dir):
    index = {}
    dirs = [(base_dir, False)]
    if opts.texture_dir and os.path.isdir(opts.texture_dir):
        dirs.append((opts.texture_dir, True))
    for d, recursive in dirs:
        walker = os.walk(d) if recursive else [(d, [], os.listdir(d))]
        for root, _dirs, files in walker:
            for fn in files:
                stem, ext = os.path.splitext(fn)
                if ext.lower() in (".png", ".dds", ".tga", ".jpg", ".jpeg", ".tif"):
                    index.setdefault((stem.lower(), ext.lower()), os.path.join(root, fn))
    return index


def _find_texture(index, fox_filename):
    stem = os.path.splitext(os.path.basename(fox_filename))[0].lower()
    for ext in (".png", ".dds", ".tga", ".jpg", ".jpeg", ".tif"):
        p = index.get((stem, ext))
        if p:
            return p
    return None


def _make_material(mi, opts, tex_index, warnings):
    mat = bpy.data.materials.new(mi.name or "FoxMaterial")
    try:
        mat.use_nodes = True          # no-op / deprecated on newest versions
    except Exception:
        pass
    mat["fox_material"] = mi.material_name
    mat["fox_material_type"] = mi.material_type
    for pname, fname, path in mi.textures:
        mat["tex_" + pname] = path + fname
    for pname, vec in mi.vectors:
        mat["vec_" + pname] = [float(x) for x in vec]

    if not (opts.load_textures and tex_index and mat.node_tree):
        return mat
    try:
        nt = mat.node_tree
        bsdf = next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if bsdf is None:
            return mat
        for pname, fname, _path in mi.textures:
            found = _find_texture(tex_index, fname)
            if not found:
                continue
            low = pname.lower()
            if low.startswith("base_tex"):
                node = nt.nodes.new("ShaderNodeTexImage")
                node.image = bpy.data.images.load(found, check_existing=True)
                node.label = pname
                nt.links.new(node.outputs["Color"], bsdf.inputs["Base Color"])
            elif low.startswith("normalmap_tex"):
                node = nt.nodes.new("ShaderNodeTexImage")
                node.image = bpy.data.images.load(found, check_existing=True)
                node.image.colorspace_settings.name = "Non-Color"
                node.label = pname
                nmap = nt.nodes.new("ShaderNodeNormalMap")
                nt.links.new(node.outputs["Color"], nmap.inputs["Color"])
                nt.links.new(nmap.outputs["Normal"], bsdf.inputs["Normal"])
    except Exception as e:  # texture hookup is best-effort only
        warnings.append(f"{mi.name}: could not build texture nodes ({e})")
    return mat


# ------------------------------------------------------------------ armature
def _build_armature(context, model, name, opts, collection):
    arm_data = bpy.data.armatures.new(name)
    arm_obj = bpy.data.objects.new(name, arm_data)
    collection.objects.link(arm_obj)

    heads = _convert(np.array([b.world_pos for b in model.bones]), opts) * opts.global_scale
    children = {}
    for i, b in enumerate(model.bones):
        if b.parent is not None:
            children.setdefault(b.parent, []).append(i)

    view_layer = context.view_layer
    for o in view_layer.objects:
        o.select_set(False)
    view_layer.objects.active = arm_obj
    arm_obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    min_len = 0.05 * opts.global_scale
    names = []
    try:
        ebones = []
        for i, b in enumerate(model.bones):
            eb = arm_data.edit_bones.new(b.name or f"bone_{i}")
            head = Vector(heads[i].tolist())
            eb.head = head
            tail = None
            kids = children.get(i, [])
            if kids:
                avg = Vector(np.mean(heads[kids], axis=0).tolist())
                if (avg - head).length > min_len * 0.1:
                    tail = avg
            eb.tail = tail if tail is not None else head + Vector((0.0, 0.0, min_len))
            ebones.append(eb)
            names.append(eb.name)
        for i, b in enumerate(model.bones):
            if b.parent is not None and b.parent < len(ebones) and b.parent != i:
                ebones[i].parent = ebones[b.parent]
    finally:
        bpy.ops.object.mode_set(mode="OBJECT")
    return arm_obj, names


# ---------------------------------------------------------------- mesh object
def _make_mesh_object(model, mesh, md, name, opts, material):
    verts = _convert(md.positions, opts) * opts.global_scale
    faces = md.faces if opts.mirror else md.faces[:, ::-1]
    # (mirroring flips handedness, which already flips the winding to match)

    me = bpy.data.meshes.new(name)
    me.from_pydata(verts.tolist(), [], faces.tolist())
    me.update()

    loops = faces.ravel()
    for i, uv in enumerate(md.uvs):
        uv = uv.copy()
        if opts.flip_v:
            uv[:, 1] = 1.0 - uv[:, 1]
        layer = me.uv_layers.new(name="UVMap" if i == 0 else f"UVMap.{i}")
        layer.data.foreach_set("uv", np.ascontiguousarray(uv[loops], np.float32).ravel())

    if md.colors is not None:
        attr = me.color_attributes.new(name="Color", type="FLOAT_COLOR", domain="POINT")
        attr.data.foreach_set("color", np.ascontiguousarray(md.colors, np.float32).ravel())

    if md.normals is not None and opts.custom_normals:
        n = _convert(md.normals, opts)
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        n = np.where(ln > 1e-6, n / np.maximum(ln, 1e-6), np.array([0, 0, 1], np.float32))
        try:
            me.normals_split_custom_set_from_vertices(n.tolist())
        except Exception:
            for p in me.polygons:
                p.use_smooth = True
    else:
        for p in me.polygons:
            p.use_smooth = True

    if material is not None:
        me.materials.append(material)

    obj = bpy.data.objects.new(name, me)
    obj["fox_mesh_index"] = mesh.index
    obj["fox_alpha_enum"] = mesh.alpha
    obj["fox_shadow_enum"] = mesh.shadow
    obj["fox_mesh_flags"] = mesh.flags

    return obj



# --------------------------------------------------------------------- skinning
def _skin_object(model, mesh, md, obj, bone_names, report):
    """Create vertex groups. Returns True if any weights were assigned."""
    tag = f"mesh {mesh.index}"
    if md.weights is None or md.bone_indices is None:
        report.append(f"{tag}: no bone weights/indices in vertex data "
                      f"(elements: {[(u, f) for u, f, *_ in md.elements]})")
        return False
    active = md.weights > 0.0
    if not active.any():
        report.append(f"{tag}: all bone weights are zero")
        return False
    top = int(md.bone_indices[active].max())

    group = model.bone_groups[mesh.bone_group] \
        if 0 <= mesh.bone_group < len(model.bone_groups) else None
    if group is not None and top < len(group) and max(group, default=-1) < len(bone_names):
        mapping = group
    elif top < len(bone_names):
        mapping = list(range(len(bone_names)))
        report.append(f"{tag}: bone group {mesh.bone_group} unusable; "
                      "treating bone indices as direct skeleton indices")
    else:
        report.append(f"{tag}: bone index {top} out of range "
                      f"(bone group {mesh.bone_group}, {len(bone_names)} bones)")
        return False

    per_bone = {}
    for slot in range(4):
        w = md.weights[:, slot]
        li = md.bone_indices[:, slot]
        for v in np.nonzero((w > 0.0) & (li < len(mapping)))[0]:
            per_bone.setdefault(mapping[int(li[v])], []).append((int(v), float(w[v])))
    for bi in sorted(per_bone):
        vg = obj.vertex_groups.new(name=bone_names[bi])
        for v, w in per_bone[bi]:
            vg.add([v], w, "ADD")
    return bool(per_bone)

# ----------------------------------------------------------------- top level
def import_fmdl(context, filepath, opts):
    """Import one .fmdl. Returns (list_of_objects, list_of_warnings)."""
    model = P.load(filepath)
    warnings = list(model.warnings)
    stem = os.path.splitext(os.path.basename(filepath))[0]
    base_dir = os.path.dirname(filepath)

    if context.object and context.object.mode != "OBJECT":
        try:
            bpy.ops.object.mode_set(mode="OBJECT")
        except Exception:
            pass

    root = bpy.data.collections.new(stem)
    context.scene.collection.children.link(root)

    # nested collections mirroring the Fox mesh-group hierarchy
    group_colls = {}

    def coll_for(gid):
        if gid in group_colls:
            return group_colls[gid]
        if gid is None or not 0 <= gid < len(model.mesh_groups):
            return root
        gname, parent, _inv = model.mesh_groups[gid]
        c = bpy.data.collections.new(gname or f"group_{gid}")
        (coll_for(parent) if parent not in (None, gid) else root).children.link(c)
        group_colls[gid] = c
        return c

    tex_index = {}
    if opts.load_textures:
        try:
            tex_index = _build_texture_index(opts, base_dir)
        except OSError:
            pass
    materials = {}

    arm_obj, bone_names = None, []
    if opts.import_armature and model.bones:
        arm_obj, bone_names = _build_armature(context, model, stem + "_armature", opts, root)

    objects = [arm_obj] if arm_obj else []
    for mesh in model.meshes:
        try:
            md = model.read_mesh(mesh)
        except P.FmdlError as e:
            warnings.append(f"mesh {mesh.index}: {e}")
            continue
        for w in md.warnings:
            warnings.append(f"mesh {mesh.index}: {w}")

        mi_idx = mesh.material_instance
        mi = model.material_instances[mi_idx] \
            if 0 <= mi_idx < len(model.material_instances) else None
        if mi is not None and mi_idx not in materials:
            materials[mi_idx] = _make_material(mi, opts, tex_index, warnings)
        material = materials.get(mi_idx)

        label = f"{mesh.index:02d}_{mi.name}" if mi else f"{mesh.index:02d}"
        target = coll_for(mesh.group if mesh.group >= 0 else None)

        lod_variants = [(0, md)]
        if opts.import_lods:
            ranges = model.lod_ranges(mesh)
            for lod in range(1, len(ranges)):
                try:
                    lod_variants.append((lod, model.read_mesh(mesh, lod=lod)))
                except P.FmdlError as e:
                    warnings.append(f"mesh {mesh.index} LOD{lod}: {e}")

        for lod, data in lod_variants:
            name = label if lod == 0 else f"{label}_LOD{lod}"
            obj = _make_mesh_object(model, mesh, data, name, opts, material)
            obj["fox_lod"] = lod
            target.objects.link(obj)
            skinned = False
            if arm_obj is not None:
                rep = warnings if lod == 0 else []
                skinned = _skin_object(model, mesh, data, obj, bone_names, rep)
            if lod > 0:
                obj.hide_set(True)
                obj.hide_render = True
            if arm_obj is not None:
                obj.parent = arm_obj
                if skinned or data.weights is not None:
                    mod = obj.modifiers.new("Armature", "ARMATURE")
                    mod.object = arm_obj
            objects.append(obj)

    return objects, warnings
