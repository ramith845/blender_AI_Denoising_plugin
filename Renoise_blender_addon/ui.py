import bpy
from bpy.types import Panel

class DENOISE_PT_panel(Panel):
    bl_label = "AI Denoise"
    bl_idname = "RENDER_PT_denoise_image"
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = 'Denoise Render'
    # bl_context = ""
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        if context.workspace is not None and context.workspace.name == 'Rendering':
            return True
        image = getattr(context.space_data, "image", None)
        return image is not None and image.type == 'RENDER_RESULT'

    # def draw_header(self, context):
    #     layout = self.layout
    #     layout.label(text="My Denoise Panel")

    def draw(self, context):
        layout = self.layout

        # layout.label(text="R Denoise")
        layout.operator('render.denoise', text="Denoise Image")
        
def register():
    bpy.utils.register_class(DENOISE_PT_panel)

def unregister():
    bpy.utils.unregister_class(DENOISE_PT_panel)