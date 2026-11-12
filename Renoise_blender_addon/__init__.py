import bpy
# import bmesh
from . import operator, ui


def register():
    operator.register()
    ui.register()

def unregister():
    ui.unregister()
    operator.unregister()

if __name__ == "__main__":
    register()