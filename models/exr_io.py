"""Read and write the dataset's EXR files with OpenImageIO (pip package `OpenImageIO`).

Same library the Blender scripts in data/ use, so files round-trip exactly.
"""

import numpy as np
import OpenImageIO as oiio


def read_exr(path):
    """Read an EXR as float32 numpy, shape (height, width, channels).

    Half-float files are converted to float32. Raises if the file can't be read,
    instead of returning None like OpenImageIO does."""
    inp = oiio.ImageInput.open(str(path))
    if inp is None:
        raise OSError(f"can't open {path}: {oiio.geterror()}")
    spec = inp.spec()
    pixels = inp.read_image(0, 0, 0, spec.nchannels, oiio.FLOAT)
    error = inp.geterror()
    inp.close()
    if pixels is None:
        raise OSError(f"can't read pixels of {path}: {error}")
    if pixels.ndim == 2:  # single channel, e.g. depth
        pixels = pixels[..., None]
    return pixels


def write_exr(path, pixels, half=True):
    """Write float numpy (H, W, 3) as a lossless (ZIP) RGB EXR, half or full float."""
    pixels = np.ascontiguousarray(pixels, dtype=np.float32)  # OIIO needs plain row order
    height, width, channels = pixels.shape
    spec = oiio.ImageSpec(width, height, channels, oiio.HALF if half else oiio.FLOAT)
    spec.attribute("compression", "zip")
    out = oiio.ImageOutput.create(str(path))
    if out is None or not out.open(str(path), spec):
        raise OSError(f"can't write {path}: {oiio.geterror()}")
    ok = out.write_image(pixels)
    error = out.geterror()
    out.close()
    if not ok:
        raise OSError(f"can't write {path}: {error}")


def write_png(path, pixels):
    """Write a display image (H, W, 3) with values 0..1 as an 8-bit PNG."""
    pixels = np.ascontiguousarray(np.clip(pixels, 0, 1) * 255 + 0.5, dtype=np.uint8)
    height, width, channels = pixels.shape
    out = oiio.ImageOutput.create(str(path))
    if out is None or not out.open(str(path), oiio.ImageSpec(width, height, channels, oiio.UINT8)):
        raise OSError(f"can't write {path}: {oiio.geterror()}")
    out.write_image(pixels)
    out.close()
