import os
import tempfile

import bpy
import numpy as np
from bpy.types import Operator


class RENDER_OT_denoise(Operator):
    bl_idname = "render.denoise"
    bl_label = "Denoise Image"
    bl_description = "Convert the Render Result to a new image (placeholder: black and white)"

    @classmethod
    def poll(cls, context):
        image = getattr(context.space_data, "image", None)
        return image is not None and image.type == 'RENDER_RESULT'

    def execute(self, context):
        render_result = context.space_data.image

        # Render Result pixels can't be read reliably, so save it to disk and reload it.
        tmp_path = os.path.join(tempfile.gettempdir(), "denoise_render_result.png")
        render_result.save_render(tmp_path)
        source = bpy.data.images.load(tmp_path)

        try:
            width, height = source.size
            pixels = np.empty(width * height * 4, dtype=np.float32)
            source.pixels.foreach_get(pixels)
            pixels = pixels.reshape(-1, 4)

            gray = pixels[:, :3].mean(axis=1)
            pixels[:, 0] = gray
            pixels[:, 1] = gray
            pixels[:, 2] = gray

            result = bpy.data.images.new("Denoised", width, height, alpha=True)
            result.pixels.foreach_set(pixels.ravel())
            result.update()
        finally:
            bpy.data.images.remove(source)
            os.remove(tmp_path)

        context.space_data.image = result
        self.report({'INFO'}, "Created image 'Denoised'")
        return {'FINISHED'}


def register():
    bpy.utils.register_class(RENDER_OT_denoise)


def unregister():
    bpy.utils.unregister_class(RENDER_OT_denoise)
