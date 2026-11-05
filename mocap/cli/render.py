"""`render` command: PNG / MP4 of the model's own mesh (the GUI Mesh view geometry: SMPL-X, or SMPL-H for FlowHMR), original vs corrected.

Views: `camera` (composited over the video frame; ExMR with the camera `review` projects joints with: one fixed camera for a static
camera, per frame following each layer's correction for a moving one; FlowHMR / GVHMR with the GUI camera view's saved camera),
`front` / `side` (virtual cameras from the body's heading) and `orbit` (close views around each self-contact).
A small numpy z-buffer rasteriser draws the mesh, so it runs headless without a GPU. numpy/torch/cv2 load lazily.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .pipeline import MODELS, CliError, current_kind, read_json, resolve_project

EXIT_USAGE = 2
VIEWS = ('camera', 'front', 'side', 'orbit')
LONG_SIDE = 900
BODY_PANEL = (600, 900)          # width, height of front / side panels
ORBIT_TILE = 420
ORBIT_AZIMUTHS = (0.0, 90.0, 180.0, 270.0)
ORBIT_ELEVATION = 20.0
ORBIT_DISTANCE = 1.1             # metres from the contact centre
FOV = 35.0
ALPHA = 0.62                     # mesh opacity over the video
MARGIN_FRAMES = 10
# BGR base colours: reference layer grey-blue, result warm.
COLORS = {'reference': (205, 185, 165), 'result': (120, 175, 235)}
BACKGROUND = (38, 38, 42)
HAND_JOINTS = {'LeftHand': (20, 28), 'RightHand': (21, 43)}   # wrist, middle knuckle 1 (SMPL-X 55-joint index)

DESCRIPTION = '''Render the motion on the model's own mesh (identical to the GUI Mesh view: SMPL-X with pose correctives,
or FlowHMR's native SMPL-H) as PNG images and optionally MP4: over the video with the project's camera, from the front and
side, and in a close orbit around each self-contact. --compare puts the reference layer (original or pre-self-contact)
beside the result.'''
EPILOG = '''frames are video / result frames (0-based, inclusive), the numbers `review` reports. Without --frames the
self-contact intervals (plus 10 frames) are rendered. ExMR, FlowHMR and GVHMR projects are supported; --sheet draws the
video hand direction only for ExMR (the other models save no hand keypoints).'''


def register(commands, common):
    from ..motion_tools import RESULTS
    p = commands.add_parser('render', parents=[common], help='render original vs corrected bodies (PNG / MP4)', description=DESCRIPTION,
                            epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project', required=True, metavar='P', help='workspace root, model project or person take')
    p.add_argument('--model', choices=MODELS, help='model inside a workspace root (default: its active model)')
    p.add_argument('--result', choices=sorted(RESULTS), metavar='KIND', help='result to render (default: the project selection)')
    p.add_argument('--frames', metavar='A-B|A,B,C', help='video frames, inclusive, or a list (default: the self-contact intervals)')
    p.add_argument('--step', type=int, default=1, metavar='N', help='render every Nth frame of A-B (default 1)')
    p.add_argument('--views', default='camera,front,side,orbit', metavar='LIST', help=f'comma list of {", ".join(VIEWS)} (default: all)')
    p.add_argument('--compare', nargs='?', const='auto', choices=('auto', 'original', 'before-selfcontact'),
                   help='also draw the reference layer: original, before-selfcontact, or auto (before-selfcontact when current)')
    p.add_argument('--focus', metavar='BONE,BONE', help='orbit around these two bones instead of the self-contact tracks, and measure their distance')
    p.add_argument('--out', metavar='DIR', help='output folder (default: <model project>/render/<result>-<time>)')
    p.add_argument('--mp4', action='store_true', help='also write one MP4 per view (FFmpeg)')
    p.add_argument('--size', type=int, default=LONG_SIDE, metavar='PX', help=f'long side of the camera view in pixels (default {LONG_SIDE})')
    p.add_argument('--sheet', action='store_true',
                   help='one comparison sheet per self-contact instead: per frame the video crop, then every layer over the same crop and '
                        'in a close 3D view, with penetration depth and skin gap (--frames A-B or A,B,C; --compare defaults to original)')
    p.add_argument('--add', action='append', default=[], metavar='LABEL=FILE',
                   help='--sheet: another result file (same params format, e.g. a saved grip-off result) as a column; repeatable')
    p.set_defaults(handler=cmd_render)


# Rasteriser ---------------------------------------------------------------------------------------------------------

@dataclass
class Camera:
    rotation: object    # (3, 3) world -> camera, OpenCV axes (+X right, +Y down, +Z forward)
    translation: object  # (3,)
    K: object           # (3, 3) pixels
    width: int
    height: int


def project(camera, points):
    """World (V, 3) -> (pixels (V, 2), camera depth (V,))."""
    import numpy as np
    cam = points @ np.asarray(camera.rotation).T + camera.translation
    z = cam[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = np.stack([camera.K[0, 0] * cam[:, 0] / z + camera.K[0, 2], camera.K[1, 1] * cam[:, 1] / z + camera.K[1, 2]], 1)
    return uv, z


def look_at(eye, target, width, height, fov=FOV, up=(0.0, 1.0, 0.0)):
    """Perspective camera at eye looking at target; world Y is up, the image keeps it up."""
    import numpy as np
    eye, target = np.asarray(eye, np.float64), np.asarray(target, np.float64)
    forward = target - eye
    forward /= max(np.linalg.norm(forward), 1e-9)
    right = np.cross(forward, np.asarray(up, np.float64))
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(forward, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    rotation = np.stack([right, down, forward])
    focal = 0.5 * height / np.tan(np.radians(fov) / 2)
    K = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]])
    return Camera(rotation, -rotation @ eye, K, int(width), int(height))


def vertex_normals(vertices, faces):
    import numpy as np
    tri = vertices[faces]
    face = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normals = np.zeros_like(vertices)
    for k in range(3):
        np.add.at(normals, faces[:, k], face)
    return normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)


def shading(camera, vertices, normals):
    """Per-vertex brightness 0..1: ambient plus a headlight and a soft key light from above."""
    import numpy as np
    cam = vertices @ np.asarray(camera.rotation).T + camera.translation
    n = normals @ np.asarray(camera.rotation).T
    view = -cam / np.maximum(np.linalg.norm(cam, axis=1, keepdims=True), 1e-9)
    key = np.array([-0.3, -0.8, -0.5])
    key /= np.linalg.norm(key)
    head = np.clip(np.sum(n * view, 1), 0, 1)
    top = np.clip(n @ key, 0, 1)
    return 0.22 + 0.6 * head + 0.25 * top


def rasterize(uv, depth, faces, shade, width, height, near=0.05, budget=3_000_000):
    """Z-buffer triangles: (brightness image (H, W) float, coverage mask (H, W), inverse depth (H, W)).

    Barycentrics come from screen-space edge functions; depth and shade are interpolated perspective-correctly."""
    import numpy as np
    zinv = np.zeros(height * width)
    value = np.zeros(height * width)
    tri, tz = uv[faces], depth[faces]
    ok = (tz > near).all(1) & np.isfinite(tri).all((1, 2))
    x0 = np.floor(tri[..., 0].min(1)).astype(np.int64)
    x1 = np.ceil(tri[..., 0].max(1)).astype(np.int64)
    y0 = np.floor(tri[..., 1].min(1)).astype(np.int64)
    y1 = np.ceil(tri[..., 1].max(1)).astype(np.int64)
    ok &= (x1 >= 0) & (y1 >= 0) & (x0 < width) & (y0 < height)
    x0, y0 = np.clip(x0, 0, width - 1), np.clip(y0, 0, height - 1)
    x1, y1 = np.clip(x1, 0, width - 1), np.clip(y1, 0, height - 1)
    area = ((tri[:, 1, 0] - tri[:, 0, 0]) * (tri[:, 2, 1] - tri[:, 0, 1]) - (tri[:, 1, 1] - tri[:, 0, 1]) * (tri[:, 2, 0] - tri[:, 0, 0]))
    ok &= np.abs(area) > 1e-9
    counts = np.where(ok, (x1 - x0 + 1) * (y1 - y0 + 1), 0)
    ids = np.flatnonzero(ok)
    if not len(ids):
        return value.reshape(height, width), zinv.reshape(height, width) > 0, zinv.reshape(height, width)
    cum = np.cumsum(counts[ids])
    start = 0
    while start < len(ids):
        stop = int(np.searchsorted(cum, (cum[start - 1] if start else 0) + budget, 'right'))
        stop = max(stop, start + 1)
        chunk = ids[start:stop]
        n = counts[chunk]
        owner = np.repeat(chunk, n)
        offset = np.arange(int(n.sum())) - np.repeat(np.cumsum(n) - n, n)
        w = (x1 - x0 + 1)[owner]
        px = x0[owner] + offset % w
        py = y0[owner] + offset // w
        cx, cy = px + 0.5, py + 0.5
        a, b, c = tri[owner, 0], tri[owner, 1], tri[owner, 2]
        s = area[owner]
        w0 = ((b[:, 0] - cx) * (c[:, 1] - cy) - (b[:, 1] - cy) * (c[:, 0] - cx)) / s
        w1 = ((c[:, 0] - cx) * (a[:, 1] - cy) - (c[:, 1] - cy) * (a[:, 0] - cx)) / s
        w2 = 1 - w0 - w1
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if inside.any():
            owner, w0, w1, w2 = owner[inside], w0[inside], w1[inside], w2[inside]
            lin = (py * width + px)[inside]
            iz = w0 / tz[owner, 0] + w1 / tz[owner, 1] + w2 / tz[owner, 2]
            sh = (w0 * shade[faces[owner, 0]] / tz[owner, 0] + w1 * shade[faces[owner, 1]] / tz[owner, 1] +
                  w2 * shade[faces[owner, 2]] / tz[owner, 2]) / iz
            order = np.lexsort((-iz, lin))
            lin, iz, sh = lin[order], iz[order], sh[order]
            first = np.r_[True, lin[1:] != lin[:-1]]
            lin, iz, sh = lin[first], iz[first], sh[first]
            closer = iz > zinv[lin]
            zinv[lin[closer]] = iz[closer]
            value[lin[closer]] = sh[closer]
        start = stop
    return value.reshape(height, width), zinv.reshape(height, width) > 0, zinv.reshape(height, width)


def draw_body(camera, vertices, faces, normals, color, background=None, alpha=1.0):
    """BGR uint8 image of one body; over `background` (same size) with `alpha`, else on a flat background."""
    import cv2
    import numpy as np
    uv, z = project(camera, vertices)
    shade = shading(camera, vertices, normals)
    value, mask, _ = rasterize(uv, z, faces, shade, camera.width, camera.height)
    if background is None:
        image = np.empty((camera.height, camera.width, 3), np.float64)
        image[:] = BACKGROUND
    else:
        image = background.astype(np.float64).copy()
    body = np.clip(value[..., None] * np.asarray(color, np.float64)[None, None] * 1.15, 0, 255)
    image[mask] = (1 - alpha) * image[mask] + alpha * body[mask]
    image = np.clip(image, 0, 255).astype(np.uint8)
    if background is not None:
        edge = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)).astype(bool)
        image[edge] = color
    return image, mask


# Scene --------------------------------------------------------------------------------------------------------------

@dataclass
class Layer:
    label: str
    role: str                 # 'reference' or 'result'
    state: dict               # smplx_body.Body.pose output
    kind: str | None          # project result kind, for the GUI mesh check


def frame_list(value, count):
    """--frames A,B,C as sorted unique frames, or None for any other value."""
    if not value or ',' not in str(value):
        return None
    try:
        frames = sorted({int(part) for part in str(value).split(',') if part.strip()})
    except ValueError:
        raise CliError('--frames expects A-B or A,B,C video frame numbers', EXIT_USAGE) from None
    if not frames or frames[0] < 0 or frames[-1] >= count:
        raise CliError(f'--frames {value}: frames must lie in 0-{count - 1}', EXIT_USAGE)
    return frames


def parse_frames(value, count):
    if value is None:
        return None
    try:
        first, last = (int(part) for part in str(value).split('-', 1))
    except ValueError:
        raise CliError('--frames expects A-B or A,B,C video frame numbers, e.g. 380-420', EXIT_USAGE) from None
    if first < 0 or last < first or last >= count:
        raise CliError(f'--frames {value}: expected 0 <= A <= B <= {count - 1}', EXIT_USAGE)
    return first, last


def contact_tracks(project, kind, frames, fps):
    """Enabled self-contact tracks of the result's base, [] when none or unreadable (with a note)."""
    from ..motion_tools import base_result
    try:
        from .. import self_contact_store as store
    except ImportError:
        return [], 'self-contact store unavailable'
    try:
        tracks, _, warning = store.read(project, base_result(kind), {'frames': frames, 'fps': fps})
    except (OSError, ValueError) as error:
        return [], f'cannot read self-contacts: {error}'
    return [t for t in tracks if t.get('enabled', True)], warning


def bone_joint(name):
    from .blender_joints import BONES
    for side in ('Left', 'Right'):
        if name == side + 'Hand':
            return HAND_JOINTS[name][0]
    return BONES.index(name)


def contact_point(state, i, bone):
    """World point of a contact end: the palm (wrist / middle knuckle midpoint) for hands, else the bone's joint."""
    if bone in HAND_JOINTS:
        wrist, knuckle = HAND_JOINTS[bone]
        return 0.5 * (state['joints'][i, wrist] + state['joints'][i, knuckle])
    return state['joints'][i, bone_joint(bone)]


def heading(states, frames):
    """Mean horizontal body forward (unit (3,)) over the frames: pelvis +Z rotated by its world rotation."""
    import numpy as np
    total = np.zeros(3)
    for state in states:
        forward = state['world'][frames, 0] @ np.array([0.0, 0.0, 1.0])
        forward[:, 1] = 0
        total += (forward / np.maximum(np.linalg.norm(forward, axis=1, keepdims=True), 1e-9)).sum(0)
    if np.linalg.norm(total) < 1e-6:
        return np.array([0.0, 0.0, 1.0])
    return total / np.linalg.norm(total)


def rotate_y(vector, degrees):
    import numpy as np
    a = np.radians(degrees)
    c, s = np.cos(a), np.sin(a)
    return np.array([c * vector[0] + s * vector[2], vector[1], -s * vector[0] + c * vector[2]])


def labels(image, lines, origin=(8, 22), scale=0.55):
    from .review import text
    text(image, lines, origin, scale)
    return image


def mark_contacts(image, camera, layer, i, tracks, offset=(0.0, 0.0)):
    """Red line effector -> target with the 3D distance, for every track active on this frame."""
    import cv2
    import numpy as np
    frame = int(layer['frame'])
    for track in tracks:
        if not any(a - MARGIN_FRAMES <= frame < b + MARGIN_FRAMES for a, b in track['ranges']):
            continue
        a = contact_point(layer['state'], i, track['effector'])
        b = contact_point(layer['state'], i, track['target'])
        (pa, pb), _ = project(camera, np.stack([a, b]))
        if not (np.isfinite(pa).all() and np.isfinite(pb).all()):
            continue
        pa, pb = (tuple(int(round(v + o)) for v, o in zip(p, offset)) for p in (pa, pb))
        active = any(a0 <= frame < b0 for a0, b0 in track['ranges'])
        color = (40, 40, 255) if active else (40, 160, 255)
        cv2.line(image, pa, pb, color, 2, cv2.LINE_AA)
        for p in (pa, pb):
            cv2.circle(image, p, 4, color, -1, cv2.LINE_AA)
        labels(image, [f"{track.get('id', '')} {np.linalg.norm(a - b) * 100:.1f} cm"], (pb[0] + 6, pb[1] - 6), 0.45)


def side_by_side(panels):
    import numpy as np
    height = max(p.shape[0] for p in panels)
    padded = [np.pad(p, ((0, height - p.shape[0]), (0, 0), (0, 0))) for p in panels]
    gap = np.full((height, 4, 3), 255, np.uint8)
    out = padded[0]
    for p in padded[1:]:
        out = np.concatenate([out, gap, p], 1)
    return out


def even(image):
    import numpy as np
    h, w = image.shape[:2]
    return np.pad(image, ((0, h % 2), (0, w % 2), (0, 0)))


class Movie:
    """FFmpeg rgb24 pipe -> H.264 MP4 (frames padded to even sizes; the first frame fixes the size)."""

    def __init__(self, path, fps):
        self.path, self.fps, self.process, self.size = Path(path), fps, None, None

    def add(self, image):
        import numpy as np
        image = even(image)
        if self.process is None:
            from ..video import executable
            self.size = image.shape[:2]
            command = [executable('ffmpeg'), '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                       '-s', f'{self.size[1]}x{self.size[0]}', '-framerate', format(self.fps, '.12g'), '-i', 'pipe:0',
                       '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '18', '-pix_fmt', 'yuv420p', str(self.path)]
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE,
                                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if image.shape[:2] != self.size:
            image = np.pad(image, ((0, max(0, self.size[0] - image.shape[0])), (0, max(0, self.size[1] - image.shape[1])), (0, 0)))
            image = image[:self.size[0], :self.size[1]]
        self.process.stdin.write(np.ascontiguousarray(image).tobytes())

    def close(self):
        if self.process is None:
            return None
        self.process.stdin.close()
        error = self.process.stderr.read().decode('utf-8', 'replace')
        if self.process.wait():
            raise CliError(f'FFmpeg could not write {self.path.name}: {error.strip()[-400:]}')
        return str(self.path)


# Views --------------------------------------------------------------------------------------------------------------

def layer_cameras(obs, project, layers, kind):
    """The review Observation per layer: a moving camera follows each layer's own corrected root (extra layers use `kind`'s)."""
    return [obs.follow(project, layer.kind or kind) for layer in layers]


@dataclass
class SavedCamera:
    """A FlowHMR / GVHMR take's camera as the GUI camera view projects with (`camera_tracks.load`), shaped like review's Observation."""
    rotation: object        # (F, 3, 3) world -> camera, OpenCV axes
    translation: object     # (F, 3)
    K: object               # (3, 3) pixels of `video`
    video: Path
    note: str
    keypoints: object = None   # no COCO-63 keypoints: the sheet leaves out the video hand direction

    def follow(self, project, kind):
        return self

    def describe(self):
        return self.note


def video_size(path):
    import cv2
    capture = cv2.VideoCapture(str(path))
    try:
        return int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        capture.release()


def saved_camera(project, model, count):
    """SavedCamera of a FlowHMR / GVHMR take: the GUI's own camera track, read without inference or refitting to the body."""
    import numpy as np
    from .. import camera_tracks
    project = Path(project).resolve()
    cache = {}
    try:
        if model == 'gvhmr':
            # The GUI reads this paired camera in a worker process; the CLI already runs in the shared runtime, so read it here.
            from ..camera_reference import reference
            cache[str(project)] = reference(project)
        camera = camera_tracks.load(project, reference_cache=cache)['cameras'][0]
    except (OSError, ValueError, KeyError, RuntimeError, MemoryError) as error:
        raise CliError(f'The camera view needs the saved {model} camera, which could not be read: {error}') from None
    matrices = camera['matrices']
    if len(matrices) not in (1, count):
        raise CliError(f'The saved {model} camera covers {len(matrices)} frames; the result has {count}.')
    matrices = np.broadcast_to(matrices, (count, 4, 4))
    video = project / 'input.mp4'
    if not video.is_file():
        from ..project_storage import source_video
        video = Path(source_video(project))
    K = np.asarray(camera['K'], np.float64).copy()
    width, height = video_size(video)
    if width and height and (width, height) != tuple(camera['size']):
        K[0] *= width / camera['size'][0]
        K[1] *= height / camera['size'][1]
    fixed = len(camera['matrices']) == 1 or np.allclose(matrices, matrices[:1])
    note = f'{model} saved camera (the GUI camera view): ' + ('one fixed camera' if fixed else 'per-frame rotation about a fixed centre')
    return SavedCamera(np.array(matrices[:, :3, :3]), np.array(matrices[:, :3, 3]), K, video, note)


def load_cameras(project, model, kind, count):
    """The camera of the camera view and sheet: ExMR's review Observation, else the SavedCamera."""
    if model == 'exmr':
        from .review import load_exmr
        return load_exmr(project, kind)
    return saved_camera(project, model, count)


class Renderer:
    def __init__(self, body, layers, frames, tracks, obs, size, focus):
        """`obs`: one review Observation per layer (camera view only)."""
        import numpy as np
        self.body, self.layers, self.frames, self.tracks, self.obs = body, layers, frames, tracks, obs
        self.faces = body.faces
        self.size = size
        states = [layer.state for layer in layers]
        self.forward = heading(states, frames)
        self.left = rotate_y(self.forward, 90.0)
        # One distance for the whole run keeps the zoom steady in a movie. The front / side panels share one camera centred
        # between the layers, so the extent is taken about that centre: a correction that moved the root still fits.
        centre = np.mean([state['joints'][frames, :22].mean(1) for state in states], 0)[:, None]
        extent = max(float(np.abs(state['joints'][frames, :22] - centre).max()) + 0.25 for state in states)
        self.distance = extent / np.tan(np.radians(FOV) / 2) * 1.05
        self.focus = focus
        self.cache = {}

    def mesh(self, layer, frame):
        key = (id(layer), frame)
        if key not in self.cache:
            vertices = self.body.vertices(layer.state, frame)
            self.cache = {k: v for k, v in self.cache.items() if k[1] == frame}
            self.cache[key] = (vertices, vertex_normals(vertices, self.faces))
        return self.cache[key]

    def info(self, layer, frame):
        return dict(state=layer.state, frame=frame)

    def camera_view(self, frame, video):
        import cv2
        import numpy as np
        scale = self.size / max(video.shape[:2])
        width, height = int(round(video.shape[1] * scale)), int(round(video.shape[0] * scale))
        background = cv2.resize(video, (width, height), interpolation=cv2.INTER_AREA)
        panels = []
        for layer, obs in zip(self.layers, self.obs):
            K = np.asarray(obs.K, np.float64).copy()
            K[:2] *= scale
            camera = Camera(obs.rotation[frame], obs.translation[frame], K, width, height)
            vertices, normals = self.mesh(layer, frame)
            image, _ = draw_body(camera, vertices, self.faces, normals, COLORS[layer.role], background, ALPHA)
            mark_contacts(image, camera, self.info(layer, frame), frame, self.tracks)
            labels(image, [f'{layer.label}', f'camera view, frame {frame}'])
            panels.append(image)
        return side_by_side(panels)

    def body_view(self, frame, name):
        import numpy as np
        width, height = BODY_PANEL
        direction = self.forward if name == 'front' else self.left
        centre = np.mean([layer.state['joints'][frame, :22].mean(0) for layer in self.layers], 0)
        camera = look_at(centre + direction * self.distance + np.array([0, 0.15, 0]), centre, width, height)
        panels = []
        for layer in self.layers:
            vertices, normals = self.mesh(layer, frame)
            image, _ = draw_body(camera, vertices, self.faces, normals, COLORS[layer.role])
            mark_contacts(image, camera, self.info(layer, frame), frame, self.tracks)
            labels(image, [layer.label, f'{name} view, frame {frame}'])
            panels.append(image)
        return side_by_side(panels)

    def orbit_view(self, frame, ends):
        import numpy as np
        rows = []
        for layer in self.layers:
            # Each row centres on its own layer's contact: the reference may sit elsewhere when correction moved the root.
            centre = 0.5 * (contact_point(layer.state, frame, ends[0]) + contact_point(layer.state, frame, ends[1]))
            tiles = []
            vertices, normals = self.mesh(layer, frame)
            for azimuth in ORBIT_AZIMUTHS:
                direction = rotate_y(self.forward, azimuth)
                elevation = np.radians(ORBIT_ELEVATION)
                eye = centre + ORBIT_DISTANCE * (np.cos(elevation) * direction + np.array([0, np.sin(elevation), 0]))
                camera = look_at(eye, centre, ORBIT_TILE, ORBIT_TILE, 40.0)
                image, _ = draw_body(camera, vertices, self.faces, normals, COLORS[layer.role])
                mark_contacts(image, camera, self.info(layer, frame), frame, self.tracks)
                labels(image, [f'{layer.label}', f'{ends[0]}-{ends[1]} az {azimuth:.0f}, frame {frame}'], scale=0.45)
                tiles.append(image)
            rows.append(side_by_side(tiles))
        gap = np.full((4, rows[0].shape[1], 3), 255, np.uint8)
        out = rows[0]
        for row in rows[1:]:
            out = np.concatenate([out, gap, row], 0)
        return out


# Command ------------------------------------------------------------------------------------------------------------

def body_name(body):
    return 'SMPL-H' if getattr(body, 'representation', 'smplx') == 'smplh' else 'SMPL-X'


def geometry(body):
    if body_name(body) == 'SMPL-H':
        return 'SMPL-H neutral LBS with shape blend shapes, no pose correctives (FlowHMR SMPLMesh, the GUI Mesh view surface)'
    return 'SMPL-X neutral LBS with shape and pose correctives (GUI Mesh view)'


def load_body(session, project, model, extra=''):
    from .smplx_body import body_for
    session.info(f"Loading the {'SMPL-H' if model == 'flowhmr' else 'SMPL-X'} body{extra} ...")
    try:
        return body_for(project, model)
    except FileNotFoundError as error:
        raise CliError(str(error), 3) from None


def load_layers(project, kind, compare, body):
    from ..motion_tools import RESULTS
    from .smplx_body import load_params
    from . import compare as comparing
    from ..self_contact_store import before_current, before_path
    layers, note = [], ''
    if compare:
        against = compare
        if compare == 'auto':
            fresh = before_current(project, kind)
            against = 'before-selfcontact' if fresh else 'original'
            if not fresh and before_path(project, kind).is_file():
                note = comparing.STALE_BEFORE.format(name=before_path(project, kind).name) + ' Comparing against the original instead.'
        path, key, label, ref_kind = comparing.reference(project, kind, against)
        layers.append(Layer(f'{label} (reference)', 'reference', body.pose(load_params(path, key)), ref_kind))
    filename, key = RESULTS[kind]
    if not (project / filename).is_file():
        raise CliError(f'{project} has no {kind} result ({filename}).')
    layers.append(Layer(kind, 'result', body.pose(load_params(project / filename, key)), kind))
    return layers, note


def cmd_render(args, session):
    started = time.monotonic()
    target = resolve_project(args.project, args.model)
    if target.project is None:
        raise CliError(f'This workspace has no {target.model} motion yet. Reconstruct it first.')
    project, model = target.project, target.model
    kind = args.result or current_kind(project, model)
    if args.sheet:
        return cmd_sheet(args, session, project, model, kind, started)
    views = [v.strip() for v in args.views.split(',') if v.strip()]
    unknown = [v for v in views if v not in VIEWS]
    if unknown or not views:
        raise CliError(f'--views: unknown {", ".join(unknown) or "(empty)"}; choose from {", ".join(VIEWS)}', EXIT_USAGE)
    if args.step < 1 or args.size < 64:
        raise CliError('--step must be >= 1 and --size >= 64', EXIT_USAGE)
    state = read_json(project / 'project.json')
    fps = float(state.get('fps', 30))
    out = Path(args.out).expanduser().resolve() if args.out else project / 'render' / f'{kind}-{time.strftime("%Y%m%d-%H%M%S")}'
    summary = dict(command='render', project=str(project), model=model, result=kind, views=views, compare=args.compare, out=str(out))
    if session.dry_run:
        summary.update(dry_run=True, frames=args.frames or 'self-contact intervals')
        if not session.json:
            print(f'Would render {kind} ({", ".join(views)}) from {project} into {out}' + (f' against {args.compare}' if args.compare else ''))
        return 0, summary
    import cv2
    import numpy as np
    from .smplx_body import gui_vertices
    body = load_body(session, project, model)
    layers, compare_note = load_layers(project, kind, args.compare, body)
    if compare_note:
        session.info('Note: ' + compare_note)
    count = len(layers[-1].state['joints'])
    tracks, note = contact_tracks(project, kind, count, fps)
    note = ' '.join(filter(None, (compare_note, note)))
    listed = frame_list(args.frames, count)
    frames = (listed[0], listed[-1]) if listed else parse_frames(args.frames, count)
    if frames is None:
        spans = [r for t in tracks for r in t['ranges']]
        if not spans:
            raise CliError('No self-contact intervals to render; pass --frames A-B or A,B,C.', EXIT_USAGE)
        frames = max(0, min(a for a, _ in spans) - MARGIN_FRAMES), min(count - 1, max(b for _, b in spans) - 1 + MARGIN_FRAMES)
    wanted = listed or list(range(frames[0], frames[1] + 1, args.step))
    markers = list(tracks)
    focus = [(t.get('id') or f"{t['effector']}-{t['target']}", (t['effector'], t['target'])) for t in tracks]
    if args.focus:
        ends = [part.strip() for part in args.focus.split(',')]
        if len(ends) != 2:
            raise CliError('--focus expects two bones, e.g. RightHand,RightFoot', EXIT_USAGE)
        for bone in ends:
            try:
                bone_joint(bone)
            except ValueError:
                raise CliError(f'--focus: unknown bone {bone}', EXIT_USAGE) from None
        focus = [(f'{ends[0]}-{ends[1]}', tuple(ends))]
        markers.append(dict(id=focus[0][0], effector=ends[0], target=ends[1], ranges=[(frames[0], frames[1] + 1)]))
    if not focus:
        focus = [('Hips', ('Hips', 'Hips'))]
        if 'orbit' in views:
            note = ' '.join(filter(None, (note, 'no self-contact tracks: the orbit centres on the hips (use --focus)')))
    if 'orbit' not in views:
        focus = []
    obs = videos = camera_note = None
    if 'camera' in views:
        from .review import read_frames
        session.info('Loading cameras and video frames ...')
        obs = load_cameras(project, model, kind, count)
        videos = read_frames(obs.video, set(wanted))
        camera_note = obs.describe()
        obs = layer_cameras(obs, project, layers, kind)
    checks = {}
    for layer in layers:
        if layer.kind is None:
            continue
        reference = gui_vertices(project, layer.kind, wanted[0])
        checks[layer.kind] = ('no mesh preview' if reference is None else
                              float(np.abs(body.vertices(layer.state, wanted[0]) - reference).max()))
    renderer = Renderer(body, layers, np.asarray(wanted), markers, obs, args.size, focus)
    out.mkdir(parents=True, exist_ok=True)
    movies = {}
    written = []
    distances = []
    clock = time.monotonic()
    for number, frame in enumerate(wanted):
        images = {}
        if 'camera' in views:
            if frame not in videos:
                raise CliError(f'The video has no frame {frame}.')
            images['camera'] = renderer.camera_view(frame, videos[frame])
        for name in ('front', 'side'):
            if name in views:
                images[name] = renderer.body_view(frame, name)
        if 'orbit' in views:
            for ident, ends in focus:
                images[f'orbit-{ident}'] = renderer.orbit_view(frame, ends)
        for name, image in images.items():
            path = out / f'{name}_f{frame:04d}.png'
            cv2.imwrite(str(path), image)
            written.append(path.name)
            if args.mp4:
                movies.setdefault(name, Movie(out / f'{name}.mp4', fps / args.step)).add(image)
        row = dict(frame=frame)
        for ident, (effector, target_bone) in ((t.get('id'), (t['effector'], t['target'])) for t in markers):
            row[ident] = {layer.role: float(np.linalg.norm(contact_point(layer.state, frame, effector) - contact_point(layer.state, frame, target_bone)))
                          for layer in layers}
        distances.append(row)
        if number % 10 == 0:
            session.info(f'  frame {frame} ({number + 1}/{len(wanted)})')
    videos_written = [movie.close() for movie in movies.values()]
    seconds = round(time.monotonic() - clock, 2)
    report = dict(version=1, command='render', project=str(project), model=model, result=kind, frames=[frames[0], frames[1]],
                  **({'frame_list': listed} if listed else {'step': args.step}),
                  views=views, camera=camera_note, layers=[dict(label=l.label, role=l.role, kind=l.kind) for l in layers],
                  geometry=geometry(body) + '; not the bound display skin',
                  gui_mesh_max_error_m=checks, tracks=[dict(id=t.get('id'), effector=t['effector'], target=t['target'], ranges=t['ranges'])
                                                       for t in tracks], orbit_focus=[dict(id=i, ends=list(e)) for i, e in focus],
                  contact_distance_m=distances, contact_point='palm = midpoint of wrist and middle knuckle; other bones = their joint',
                  note=note or None, images=written, movies=[v for v in videos_written if v],
                  seconds=dict(render=seconds, total=round(time.monotonic() - started, 2)))
    (out / 'render.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    session.say(f'Render: {len(written)} image(s){", " + str(len(report["movies"])) + " movie(s)" if report["movies"] else ""} in {out}')
    summary.update(report_path=str(out / 'render.json'), images=len(written), movies=report['movies'], gui_mesh_max_error_m=checks,
                   frames=report['frames'])
    return 0, summary


# Comparison sheet ---------------------------------------------------------------------------------------------------

SHEET_TILE = 340
SHEET_FRAMES = 4
SHEET_SIDE_DEG = 90.0            # close 3D view: the camera direction turned about the vertical
SHEET_DISTANCE = 0.75
SHEET_COLORS = ((205, 185, 165), (120, 175, 235), (140, 210, 140), (200, 150, 220))
GREEN, MAGENTA = (60, 220, 60), (230, 60, 230)


def spread(frames, count=SHEET_FRAMES):
    """`count` evenly spaced items of the sorted frames (all of them when fewer)."""
    import numpy as np
    if len(frames) <= count:
        return list(frames)
    return [frames[int(round(k))] for k in np.linspace(0, len(frames) - 1, count)]


def sheet_frames(value, ranges, count, automatic=()):
    """(frames, outside) of one contact's sheet. The hold is the contact's ranges, else its suggested ranges (end-exclusive).
    SHEET_FRAMES frames spread over the hold's frames, within --frames A-B when given; --frames A,B,C keeps the listed frames
    inside the hold. When no requested frame is inside it, the requested frames are drawn and listed in `outside` (rows
    marked out of the hold). None without --frames for a contact that has no hold."""
    spans = list(ranges) or list(automatic)
    held = sorted({f for a, b in spans for f in range(max(int(a), 0), min(int(b), count))})
    if not value:
        return (spread(held), []) if held else None
    listed = frame_list(value, count)
    if listed is None:
        first, last = parse_frames(value, count)
        requested = list(range(first, last + 1))
    else:
        requested = listed
    inside = sorted(set(requested) & set(held))
    if inside:
        return (inside if listed else spread(inside)), []
    picked = listed or spread(requested)
    return picked, picked


def crop_camera(obs, frame, box, tile):
    """The project camera of a frame looking at the square video crop box (x0, y0, side) at tile pixels."""
    import numpy as np
    x0, y0, side = box
    K = np.asarray(obs.K, np.float64).copy()
    K[0, 2] -= x0
    K[1, 2] -= y0
    K[:2] *= tile / side
    return Camera(obs.rotation[frame], obs.translation[frame], K, tile, tile)


def crop_video(image, box, tile):
    import cv2
    x0, y0, side = (int(round(v)) for v in box)
    padded = cv2.copyMakeBorder(image, side, side, side, side, cv2.BORDER_CONSTANT, value=BACKGROUND)
    return cv2.resize(padded[y0 + side:y0 + 2 * side, x0 + side:x0 + 2 * side], (tile, tile), interpolation=cv2.INTER_AREA)


def skin_numbers(groups, segments, vertices, effector, arm, targets):
    """(hand depth m, skin gap m or nan, arm depth m) of the effector hand and its arm segments against the target segments."""
    from ..self_contact import penetrations, signed_gap
    posed = {name: vertices[groups[name]] for name in targets}
    hand = penetrations(vertices[groups[effector]], posed, segments, list(targets))
    arm_depth = max([penetrations(vertices[groups[name]], posed, segments, list(targets))['depth'] for name in arm] or [0.0])
    return hand['depth'], signed_gap(hand), arm_depth


def hand_arrow(image, camera, start, end, color):
    import cv2
    import numpy as np
    (a, b), _ = project(camera, np.stack([start, end]))
    if np.isfinite(a).all() and np.isfinite(b).all():
        cv2.arrowedLine(image, tuple(int(round(v)) for v in a), tuple(int(round(v)) for v in b), color, 2, cv2.LINE_AA, tipLength=0.25)
        return b - a
    return None


def sheet_columns(layers):
    return ['Video'] + [layer.label for layer in layers] + [f'{layer.label} - 3D' for layer in layers]


def side_axis(obs, frame):
    """Horizontal unit vector SHEET_SIDE_DEG from the camera's viewing direction on this frame."""
    import numpy as np
    forward = np.asarray(obs.rotation[frame]).T @ np.array([0.0, 0.0, 1.0])
    forward[1] = 0
    return rotate_y(forward / max(np.linalg.norm(forward), 1e-9), SHEET_SIDE_DEG)


def hand_side(state, track, frames, obs):
    """+1 or -1: which side_axis direction the hand sits on relative to its target over the frames, so the close view looks
    at the hand rather than through the target."""
    lean = sum(float(side_axis(obs, f) @ (contact_point(state, f, track['effector']) - contact_point(state, f, track['target']))) for f in frames)
    return 1.0 if lean >= 0 else -1.0


def contact_sheet_image(body, layers, track, frames, cameras, videos, tile, outside=()):
    """Rows = frames; columns = video crop, every layer over the crop, every layer in a close 3D view; numbers per tile.

    `cameras`: one review Observation per layer; the crop follows the result (last) layer. Rows of `outside` frames are
    marked as outside the contact's hold."""
    import cv2
    import numpy as np
    obs = cameras[-1]
    sign = hand_side(layers[-1].state, track, frames, obs)
    from ..self_contact import Segments, arm_groups, parse_target, segment_names, segment_vertices
    _, side, part = parse_target(track['target'])
    h = 0 if track['effector'] == 'LeftHand' else 1
    groups = segment_vertices(body.weights)
    segments = Segments(body.faces, groups)
    targets = segment_names(part, side)
    effector, *arm = arm_groups(h, part, side)
    wrist, knuckle = HAND_JOINTS[track['effector']]
    # The close view shows only the touching limbs, so the trunk or head never hides the contact.
    shown = np.zeros(len(body.weights), bool)
    for name in [effector] + arm + targets + [n for n in (targets[0].rsplit('_', 1)[0] + '_thigh',) if n in groups]:
        shown[groups[name]] = True
    limb_faces = body.faces[shown[body.faces].all(1)]
    base = 23 + 20 * h
    rows = []
    for frame in frames:
        result = layers[-1].state
        ends = np.stack([contact_point(result, frame, track['effector']), contact_point(result, frame, track['target'])])
        others = np.stack([contact_point(layer.state, frame, track['effector']) for layer in layers])
        frame_camera = Camera(obs.rotation[frame], obs.translation[frame], obs.K, 1, 1)
        uv = project(frame_camera, np.concatenate([ends, others]))[0]
        uv = uv[np.isfinite(uv).all(1)]
        kp = obs.keypoints[frame] if obs.keypoints is not None else None
        if kp is not None:
            torso = float(np.linalg.norm(obs.keypoints[frame, 5, :2] - obs.keypoints[frame, 12, :2]))
        else:
            # No 2D keypoints: the projected model's left shoulder to right hip (COCO 5 / 12) sizes the crop instead.
            shoulder, hip = project(frame_camera, result['joints'][frame, [16, 2]])[0]
            torso = float(np.linalg.norm(shoulder - hip)) if np.isfinite([shoulder, hip]).all() else 0.0
        side_px = max(float(np.ptp(uv, 0).max()) * 1.6, 0.9 * torso, 160.0) if len(uv) else 400.0
        centre = uv.mean(0) if len(uv) else np.array(videos[frame].shape[1::-1]) / 2
        box = (centre[0] - side_px / 2, centre[1] - side_px / 2, side_px)
        video = crop_video(videos[frame], box, tile)
        point = lambda xy: tuple(int(round(v)) for v in (np.asarray(xy) - box[:2]) * tile / side_px)
        kp_dir = None
        if kp is None:
            labels(video, [f'frame {frame}', 'no saved 2D hand keypoints'], scale=0.45)
        else:
            if kp[9 + h, 2] > 0.3 and kp[base + 8, 2] > 0.3:
                cv2.arrowedLine(video, point(kp[9 + h, :2]), point(kp[base + 8, :2]), GREEN, 2, cv2.LINE_AA, tipLength=0.25)
                kp_dir = kp[base + 8, :2] - kp[9 + h, :2]
            labels(video, [f'frame {frame}', 'green: video wrist->knuckle', f'score {min(kp[9 + h, 2], kp[base + 8, 2]):.2f}'], scale=0.45)
        if frame in outside:
            cv2.rectangle(video, (1, 1), (tile - 2, tile - 2), (40, 40, 255), 3)
            labels(video, ['outside the hold'], (8, tile - 12), 0.5)
        overlays, views = [], []
        offset = sign * side_axis(obs, frame) * SHEET_DISTANCE + np.array([0.0, 0.2, 0.0])
        for k, layer in enumerate(layers):
            # Each close view centres on its own layer's contact: the reference may sit elsewhere when correction moved the root.
            centre3 = 0.5 * (contact_point(layer.state, frame, track['effector']) + contact_point(layer.state, frame, track['target']))
            close = look_at(centre3 + offset, centre3, tile, tile, 40.0)
            camera = crop_camera(cameras[k], frame, box, tile)
            vertices = body.vertices(layer.state, frame)
            normals = vertex_normals(vertices, body.faces)
            color = SHEET_COLORS[k % len(SHEET_COLORS)]
            depth, gap, arm_depth = skin_numbers(groups, segments, vertices, effector, arm, targets)
            numbers = [f'hand inside {targets[0].split("_")[-1]}: {depth * 100:.1f} cm' if depth > 0.002 else
                       'skin gap: ' + ('> 5 cm' if not np.isfinite(gap) else f'{max(gap, 0) * 100:.1f} cm')]
            if arm_depth > 0.005:
                numbers.append(f'arm inside target: {arm_depth * 100:.1f} cm')
            image, _ = draw_body(camera, vertices, body.faces, normals, color, crop_video(videos[frame], box, tile), ALPHA)
            model_dir = hand_arrow(image, camera, layer.state['joints'][frame, wrist], layer.state['joints'][frame, knuckle], MAGENTA)
            if kp_dir is not None and model_dir is not None:
                error = np.degrees(abs(np.arctan2(model_dir[0] * kp_dir[1] - model_dir[1] * kp_dir[0], model_dir @ kp_dir)))
                numbers.append(f'hand direction vs video: {error:.0f} deg')
            labels(image, numbers, scale=0.45)
            overlays.append(image)
            view, _ = draw_body(close, vertices, limb_faces, vertex_normals(vertices, limb_faces), color)
            labels(view, numbers[:2] + [f'3D, {SHEET_SIDE_DEG:.0f} deg off camera, hand side, limbs only'], scale=0.45)
            views.append(view)
        rows.append(side_by_side([video] + overlays + views))
    header = np.full((34, rows[0].shape[1], 3), 255, np.uint8)
    for k, title in enumerate(sheet_columns(layers)):
        cv2.putText(header, title, (k * (tile + 4) + 8, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    legend = np.full((30, rows[0].shape[1], 3), 255, np.uint8)
    video_key = 'green: video keypoints' if obs.keypoints is not None else 'no video keypoints saved'
    cv2.putText(legend, f"{track.get('id')}: {track['effector']} -> {track['target']}. Magenta arrow: model wrist->middle knuckle; {video_key}"
                f". Depth/gap: {body_name(body)} skin of the hand vs the {' + '.join(targets)}.", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (20, 20, 20), 1, cv2.LINE_AA)
    gap = np.full((4, rows[0].shape[1], 3), 255, np.uint8)
    out = [header]
    for row in rows:
        out += [row, gap]
    return np.concatenate(out + [legend], 0)


def cmd_sheet(args, session, project, model, kind, started):
    from ..motion_tools import RESULTS
    state = read_json(project / 'project.json')
    fps = float(state.get('fps', 30))
    out = Path(args.out).expanduser().resolve() if args.out else project / 'render' / f'{kind}-sheet-{time.strftime("%Y%m%d-%H%M%S")}'
    extra = []
    for item in args.add:
        label, sep, path = item.partition('=')
        if not sep or not Path(path).expanduser().is_file():
            raise CliError(f'--add expects LABEL=FILE with an existing file: {item}', EXIT_USAGE)
        extra.append((label, Path(path).expanduser().resolve()))
    summary = dict(command='render', sheet=True, project=str(project), model=model, result=kind, out=str(out))
    if session.dry_run:
        summary.update(dry_run=True)
        if not session.json:
            print(f'Would write one comparison sheet per self-contact of {kind} into {out}')
        return 0, summary
    import cv2
    from .review import read_frames
    from .smplx_body import load_params
    body = load_body(session, project, model, ', cameras and keypoints')
    from . import compare as comparing
    path, key, label, ref_kind = comparing.reference(project, kind, args.compare or 'original')
    layers = [Layer('Before (original)' if (args.compare or 'original') == 'original' else 'Before (' + label + ')', 'reference',
                    body.pose(load_params(path, key)), ref_kind)]
    for name, file in extra:
        try:
            layers.append(Layer(name, 'result', body.pose(load_params(file)), None))
        except KeyError as error:
            raise CliError(f'--add {file.name}: not a {body_name(body)} result like {kind} (no {error}).', EXIT_USAGE) from None
    filename, key = RESULTS[kind]
    count = len(layers[0].state['joints'])
    tracks, note = contact_tracks(project, kind, count, fps)
    if not tracks:
        raise CliError('No enabled self-contact tracks to make a sheet for.', EXIT_USAGE)
    grip = {t.get('grip', 'off') for t in tracks}
    layers.append(Layer('After' + (f' (grip {grip.pop()})' if len(grip) == 1 else ''), 'result', body.pose(load_params(project / filename, key)), kind))
    obs = layer_cameras(load_cameras(project, model, kind, count), project, layers, kind)
    if obs[-1].keypoints is None:
        note = ' '.join(filter(None, (note, f'{model} saves no COCO-63 hand keypoints, so the sheet leaves out the video hand direction '
                                            '(green arrow and angle) and sizes each crop from the model.')))
        session.info('Note: ' + note)
    plans = [(track, sheet_frames(args.frames, track['ranges'], count, track.get('automatic', ()))) for track in tracks]
    track_name = lambda track: track.get('id') or f"{track['effector']}-{track['target']}"
    skipped = [track_name(track) for track, frames in plans if frames is None]
    plans = [(track, *frames) for track, frames in plans if frames is not None]
    if skipped:
        note = ' '.join(filter(None, (note, f"Skipped {', '.join(skipped)}: no frame ranges or suggestions (pass --frames).")))
        session.info('Note: ' + note)
    outside = [track_name(track) for track, _, out_of_hold in plans if out_of_hold]
    if outside:
        note = ' '.join(filter(None, (note, f"No --frames inside the hold of {', '.join(outside)}: those rows are marked 'outside the hold'.")))
        session.info('Note: ' + note)
    if not plans:
        raise CliError('No self-contact frame ranges to make a sheet for; pass --frames A-B.', EXIT_USAGE)
    videos = read_frames(obs[0].video, {f for _, frames, _ in plans for f in frames})
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for track, frames, out_of_hold in plans:
        missing = [f for f in frames if f not in videos]
        if missing:
            raise CliError(f'The video has no frame {missing[0]}.')
        image = contact_sheet_image(body, layers, track, frames, obs, videos, SHEET_TILE, out_of_hold)
        path = out / f"sheet_{track.get('id') or 'contact'}_f{frames[0]:04d}-{frames[-1]:04d}.png"
        cv2.imwrite(str(path), image)
        written.append(str(path))
        session.info(f'  {path.name}')
    report = dict(version=1, command='render', sheet=True, project=str(project), result=kind, layers=[l.label for l in layers],
                  frames={t.get('id'): f for t, f, _ in plans}, outside_hold={t.get('id'): o for t, _, o in plans if o},
                  images=written, note=note or None, camera=obs[-1].describe(),
                  geometry=geometry(body), seconds=round(time.monotonic() - started, 2))
    (out / 'sheet.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    session.say(f'Sheet: {len(written)} image(s) in {out}')
    summary.update(images=written, report_path=str(out / 'sheet.json'))
    return 0, summary
