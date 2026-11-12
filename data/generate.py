import subprocess
from pathlib import Path

BLENDER = r"E:\blender\stable\Blender 5.2\blender.exe"
DATA_DIR = Path(__file__).resolve().parent  # this folder, wherever you launch from
CUSTOM_SCENE = DATA_DIR / "custom_scene.py"
OUT_DIR = DATA_DIR / "dataset"
FRAMES = 150  # frames per scene; try ~20 first to check the pipeline (roadmap 1.2)

# Scenes to render, listed by hand so library files (classroom/assets/*.blend) and
# test files never get rendered. Each scene needs a CamBounds Empty (see README.md).
SCENES = [
    DATA_DIR / "blendfiles" / "classroom" / "classroom.blend",
    DATA_DIR / "blendfiles" / "barbershop" / "barbershop_interior.blend",  # if it fits in 6 GB VRAM
    DATA_DIR / "blendfiles" / "benchmark" / "benchmark.blend",
    DATA_DIR / "blendfiles" / "b_281_splash" / "blender-2.81-splash.blend", 
    # DATA_DIR / "blendfiles" / "b_282_splash" / "blender-2.82-splash.blend",
    DATA_DIR / "blendfiles" / "b_283_splash" / "blender-283.blend",
    DATA_DIR / "blendfiles" / "b_310_splash" / "blender3.1-splash.blend",
    # DATA_DIR / "blendfiles" / "b_330_splash" / "blender-3.3-splash.blend",
    DATA_DIR / "blendfiles" / "b_410_splash" / "blender-4.1-splash.blend",
    DATA_DIR / "blendfiles" / "b_430_splash" / "blender-4.3-splash.blend",
    # DATA_DIR / "blendfiles" / "b_520_splash" / "blender-5.2-splash.blend",
]

# Procedural rooms made by layout.py (see README_layouts.md). Many small scenes with
# fewer frames each (roadmap 1.2: ~100 layouts x 15 frames).
LAYOUTS = sorted((DATA_DIR / "blendfiles" / "procedural").glob("layout_*.blend"))
LAYOUT_FRAMES = 15

# Blender runs args left to right:
# [BLENDER] + PRE_FILE_ARGS + [blend file] + POST_FILE_ARGS + ["--"] + SCRIPT_ARGS
# Loading the .blend overwrites render settings, so -o/-E/-F/... must come AFTER
# the file, and the render action (-f / -a) must come after those settings.
# Value True = bare flag, None/False = skipped, anything else = flag + value.

# Before the blend file.
PRE_FILE_ARGS = {
    "-b": None,  # --background: run without UI
    "--factory-startup": None,  # skip the user's startup.blend
    "-y": None,  # --enable-autoexec: allow auto-run Python in the .blend
    "-Y": None,  # --disable-autoexec (default)
    "--python-use-system-env": None,  # allow PYTHONPATH / user site-packages
    "--addons": None,  # comma list (no spaces) of add-ons to enable
    "--app-template": None,  # app template dir name, 'default' for none
    "-noaudio": None,  # force sound system to None
    "-setaudio": None,  # None, SDL, OpenAL, CoreAudio, JACK, PulseAudio, WASAPI
    "--debug-all": None,  # all debug messages
    "-d": None,  # --debug: memory error detection, etc.
    "--debug-value": None,  # <value> debug value on startup
    "--debug-cycles": None,
    "--debug-memory": None,
    "--debug-python": None,
    "--debug-gpu": None,
    "--debug-exit-on-error": None,  # exit as soon as an internal error is detected
    "--disable-crash-handler": None,
    "--disable-abort-handler": None,
    "--verbose": None,  # <verbose> debug-message verbosity
    "--log": None,  # <match> categories, e.g. "wm.*" or "*"
    "--log-level": None,  # <level> default 1, -1 logs everything
    "--log-file": None,  # <filepath> write log to a file
    "--log-show-basename": None,
    "--log-show-timestamp": None,
    "--gpu-backend": None,  # e.g. 'opengl'
    "--python-exit-code": None,  # <0..255> exit code on Python exception
}

# After the blend file, in the order Blender should process them.
POST_FILE_ARGS = {
    "-S": None,  # --scene <name>: active scene
    "-E": None,  # --engine <engine>: use '-E help' to list
    "-t": None,  # --threads <1-1024>, 0 = all cores
    "-o": None,  # --render-output <path>; // = relative to .blend, # = frame digits
    "-F": None,  # --render-format: TGA RAWTGA JPEG IRIS AVIRAW AVIJPEG PNG BMP HDR TIFF
    #   OPEN_EXR OPEN_EXR_MULTILAYER FFMPEG CINEON DPX JP2 WEBP
    "-x": None,  # --use-extension <bool>: append file extension (0/1)
    "-s": None,  # --frame-start <frame>
    "-e": None,  # --frame-end <frame>
    "-j": None,  # --frame-jump <frames>: step between rendered frames
    "-P": None,  # --python <filepath>: run a script (runs in order, before a later -f)
    "--python-text": None,  # <name> run a text block
    "--python-expr": None,  # <expression> run a Python expression
    # Render action, keep last. Use ONE of these:
    "-f": None,  # --render-frame <frame|list "1,5,9"|range "1..10">, +/- = relative
    "-a": None,  # --render-anim: start to end frame (True for bare flag)
}

# After "--": ignored by Blender, read by Cycles / your own scripts.
SCRIPT_ARGS = {
    "--cycles-device": None,  # CPU CUDA OPTIX HIP ONEAPI METAL; append +CPU for both
    "--cycles-print-stats": None,  # log render memory/time stats
    "--out-dir": None,  # custom_scene.py: dataset folder (scenes go in <out-dir>/<scene>/)
    "--frames": None,  # custom_scene.py: frames to save per scene
}


def dict_to_args(args):
    out = []
    for flag, val in args.items():
        if val is None or val is False:
            continue
        out.append(flag)
        if val is not True:
            out.append(str(val))
    return out


def build_args(file, frames):
    pre = dict(PRE_FILE_ARGS)
    post = dict(POST_FILE_ARGS)
    script = dict(SCRIPT_ARGS)

    pre["-b"] = True
    # Without this, a Python error in custom_scene.py still exits with code 0
    pre["--python-exit-code"] = 1

    post["-E"] = "CYCLES"
    # post["-o"] = "//render_"
    # post["-F"] = "PNG"
    # post["-f"] = "1"
    post["-t"] = 0  # all cores
    post["-P"] = str(CUSTOM_SCENE)

    script["--cycles-device"] = "OPTIX"
    script["--out-dir"] = str(OUT_DIR)
    script["--frames"] = frames

    cmd = [BLENDER]
    cmd += dict_to_args(pre)
    cmd.append(str(file))
    cmd += dict_to_args(post)
    script_args = dict_to_args(script)
    if script_args:
        cmd += ["--"] + script_args
    return cmd


def render_scene(blend, frames):
    """Run custom_scene.py on one .blend. Returns True if it finished cleanly."""
    log_path = blend.with_name(blend.stem + "_render.log")
    err_path = blend.with_name(blend.stem + "_render.err")
    print(f"{blend.name}: rendering, progress in {log_path}", flush=True)
    with open(log_path, "w") as log, open(err_path, "w") as err_log:
        result = subprocess.run(build_args(blend, frames), stdout=log, stderr=err_log)
    if result.returncode != 0:
        print(f"{blend.name}: FAILED (exit code {result.returncode}), see {err_path}", flush=True)
        return False
    print(f"{blend.name}: done", flush=True)
    return True


jobs = [(blend, FRAMES) for blend in SCENES] + [(blend, LAYOUT_FRAMES) for blend in LAYOUTS]
failed = []
for blend, frames in jobs:
    if not blend.exists():
        print(f"{blend}: not found, skipping", flush=True)
        failed.append(blend.name)
    elif not render_scene(blend, frames):
        failed.append(blend.name)  # keep going; re-running resumes where it stopped

print(f"{len(jobs) - len(failed)}/{len(jobs)} scenes finished", flush=True)
if failed:
    raise SystemExit(f"failed: {', '.join(failed)}")
