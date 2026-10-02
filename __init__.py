# SPDX-License-Identifier: GPL-3.0-or-later
"""Fox Engine FMDL importer (File > Import > Fox Engine Model (.fmdl))."""

import os

import bpy
from bpy.props import BoolProperty, CollectionProperty, FloatProperty, StringProperty
from bpy_extras.io_utils import ImportHelper

from . import fmdl_import, fmdl_parser


class IMPORT_SCENE_OT_fmdl(bpy.types.Operator, ImportHelper):
    """Import a Fox Engine model (.fmdl)"""
    bl_idname = "import_scene.fmdl"
    bl_label = "Import FMDL"
    bl_options = {"REGISTER", "UNDO", "PRESET"}

    filename_ext = ".fmdl"
    filter_glob: StringProperty(default="*.fmdl", options={"HIDDEN"})
    files: CollectionProperty(type=bpy.types.OperatorFileListElement,
                              options={"HIDDEN", "SKIP_SAVE"})
    directory: StringProperty(subtype="DIR_PATH", options={"HIDDEN", "SKIP_SAVE"})

    global_scale: FloatProperty(name="Scale", default=1.0, min=0.0001, max=10000.0)
    mirror: BoolProperty(
        name="Convert Handedness",
        description="Fox uses a left-handed Y-up system; mirror into Blender's "
                    "right-handed Z-up. Disable to only rotate (model will be mirrored)",
        default=True)
    flip_v: BoolProperty(name="Flip UV V", default=True,
                         description="Fox UVs have their origin at the top-left")
    custom_normals: BoolProperty(name="Custom Normals", default=True)
    import_armature: BoolProperty(name="Skeleton && Weights", default=True)
    import_lods: BoolProperty(
        name="Import Lower LODs", default=False,
        description="Also import LOD1+ as hidden objects")
    load_textures: BoolProperty(
        name="Look for Textures", default=True,
        description="Hook up textures if image files with matching names "
                    "(.png/.dds/.tga) exist next to the model or in the texture folder")
    texture_dir: StringProperty(name="Texture Folder", subtype="DIR_PATH", default="")

    def draw(self, context):
        layout = self.layout
        layout.use_property_split = True
        layout.prop(self, "global_scale")
        layout.prop(self, "mirror")
        layout.prop(self, "flip_v")
        layout.prop(self, "custom_normals")
        layout.prop(self, "import_armature")
        layout.prop(self, "import_lods")
        layout.prop(self, "load_textures")
        sub = layout.column()
        sub.active = self.load_textures
        sub.prop(self, "texture_dir")

    def execute(self, context):
        opts = fmdl_import.ImportOptions(
            global_scale=self.global_scale, mirror=self.mirror, flip_v=self.flip_v,
            custom_normals=self.custom_normals, import_armature=self.import_armature,
            import_lods=self.import_lods, load_textures=self.load_textures,
            texture_dir=bpy.path.abspath(self.texture_dir))

        paths = [os.path.join(self.directory, f.name) for f in self.files] \
            if self.files and self.directory else [self.filepath]
        total, warn = 0, []
        for path in paths:
            try:
                objs, w = fmdl_import.import_fmdl(context, path, opts)
            except (fmdl_parser.FmdlError, OSError) as e:
                self.report({"ERROR"}, f"{os.path.basename(path)}: {e}")
                return {"CANCELLED"}
            total += len(objs)
            warn += w
        for w in warn[:10]:
            self.report({"WARNING"}, w)
        if len(warn) > 10:
            self.report({"WARNING"}, f"... and {len(warn) - 10} more warnings")
        self.report({"INFO"}, f"Imported {total} object(s) from {len(paths)} file(s)")
        return {"FINISHED"}


class IO_FH_fmdl(bpy.types.FileHandler):
    bl_idname = "IO_FH_fmdl"
    bl_label = "Fox Engine Model"
    bl_import_operator = "import_scene.fmdl"
    bl_file_extensions = ".fmdl"

    @classmethod
    def poll_drop(cls, context):
        from bpy_extras.io_utils import poll_file_object_drop
        return poll_file_object_drop(context)


def menu_func_import(self, context):
    self.layout.operator(IMPORT_SCENE_OT_fmdl.bl_idname, text="Fox Engine Model (.fmdl)")


classes = (IMPORT_SCENE_OT_fmdl, IO_FH_fmdl)


def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)


def unregister():
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)
    for c in reversed(classes):
        bpy.utils.unregister_class(c)
