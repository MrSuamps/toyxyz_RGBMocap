"""`render` command: PNG / MP4 of the model's own SMPL-X mesh (the GUI Mesh view geometry), original vs corrected.

Views: `camera` (composited over the video frame with the project's per-frame camera, as `review` projects joints),
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

DESCRIPTION = '''Render the motion on the model's own SMPL-X mesh (identical to the GUI Mesh view, including pose
correctives) as PNG images and optionally MP4: over the video with the project's camera, from the front and side, and
in a close orbit around each self-contact. --compare puts the reference layer (original or pre-self-contact) beside the
result.'''
EPILOG = '''frames are video / result frames (0-based, inclusive), the numbers `review` reports. Without --frames the
self-contact intervals (plus 10 frames) are rendered. The camera view needs an ExMR project (its saved cameras).'''


def register(commands, common):
    from ..motion_tools import RESULTS
    p = commands.add_parser('render', parents=[common], help='render original vs corrected bodies (PNG / MP4)', description=DESCRIPTION,
                            epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--project', required=True, metavar='P', help='workspace root, model project or person take')
    p.add_argument('--model', choices=MODELS, help='model inside a workspace root (default: its active model)')
    p.add_argument('--result', choices=sorted(RESULTS), metavar='KIND', help='result to render (default: the project selection)')
    p.add_argument('--frames', metavar='A-B', help='video frames, inclusive (default: the self-contact intervals)')
    p.add_argument('--step', type=int, default=1, metavar='N', help='render every Nth frame (default 1)')
    p.add_argument('--views', default='camera,front,side,orbit', metavar='LIST', help=f'comma list of {", ".join(VIEWS)} (default: all)')
    p.add_argument('--compare', nargs='?', const='auto', choices=('auto', 'original', 'before-selfcontact'),
                   help='also draw the reference layer: original, before-selfcontact, or auto (before-selfcontact when present)')
    p.add_argument('--focus', metavar='BONE,BONE', help='orbit around these two bones instead of the self-contact tracks, and measure their distance')
    p.add_argument('--out', metavar='DIR', help='output folder (default: <model project>/render/<result>-<time>)')
    p.add_argument('--mp4', action='store_true', help='also write one MP4 per view (FFmpeg)')
    p.add_argument('--size', type=int, default=LONG_SIDE, metavar='PX', help=f'long side of the camera view in pixels (default {LONG_SIDE})')
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


def parse_frames(value, count):
    if value is None:
        return None
    try:
        first, last = (int(part) for part in str(value).split('-', 1))
    except ValueError:
        raise CliError('--frames expects A-B video frame numbers, e.g. 380-420', EXIT_USAGE) from None
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

class Renderer:
    def __init__(self, body, layers, frames, tracks, obs, size, focus):
        import numpy as np
        self.body, self.layers, self.frames, self.tracks, self.obs = body, layers, frames, tracks, obs
        self.faces = body.faces
        self.size = size
        states = [layer.state for layer in layers]
        self.forward = heading(states, frames)
        self.left = rotate_y(self.forward, 90.0)
        # One distance for the whole run keeps the zoom steady in a movie.
        extent = 0.0
        for state in states:
            joints = state['joints'][frames, :22]
            centre = joints.mean(1, keepdims=True)
            extent = max(extent, float(np.abs(joints - centre).max()) + 0.25)
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
        K = np.asarray(self.obs.K, np.float64).copy()
        K[:2] *= scale
        camera = Camera(self.obs.rotation[frame], self.obs.translation[frame], K, width, height)
        panels = []
        for layer in self.layers:
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
        result = self.layers[-1]
        centre = 0.5 * (contact_point(result.state, frame, ends[0]) + contact_point(result.state, frame, ends[1]))
        for layer in self.layers:
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

def load_layers(project, kind, compare, body):
    from ..motion_tools import RESULTS
    from .smplx_body import load_params
    from . import compare as comparing
    layers = []
    if compare:
        against = compare
        if compare == 'auto':
            against = 'before-selfcontact' if (project / f'{kind}_before_selfcontact.pt').is_file() else 'original'
        path, key, label, ref_kind = comparing.reference(project, kind, against)
        layers.append(Layer(f'{label} (reference)', 'reference', body.pose(load_params(path, key)), ref_kind))
    filename, key = RESULTS[kind]
    if not (project / filename).is_file():
        raise CliError(f'{project} has no {kind} result ({filename}).')
    layers.append(Layer(kind, 'result', body.pose(load_params(project / filename, key)), kind))
    return layers


def cmd_render(args, session):
    started = time.monotonic()
    target = resolve_project(args.project, args.model)
    if target.project is None:
        raise CliError(f'This workspace has no {target.model} motion yet. Reconstruct it first.')
    project, model = target.project, target.model
    kind = args.result or current_kind(project, model)
    views = [v.strip() for v in args.views.split(',') if v.strip()]
    unknown = [v for v in views if v not in VIEWS]
    if unknown or not views:
        raise CliError(f'--views: unknown {", ".join(unknown) or "(empty)"}; choose from {", ".join(VIEWS)}', EXIT_USAGE)
    if args.step < 1 or args.size < 64:
        raise CliError('--step must be >= 1 and --size >= 64', EXIT_USAGE)
    if 'camera' in views and model != 'exmr':
        raise CliError(f'The camera view needs the ExMR saved cameras; {model} is not supported yet. Use --views front,side,orbit.')
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
    from .smplx_body import Body, gui_vertices, model_path
    session.info('Loading the SMPL-X body ...')
    body = Body(model_path(project))
    layers = load_layers(project, kind, args.compare, body)
    count = len(layers[-1].state['joints'])
    tracks, note = contact_tracks(project, kind, count, fps)
    frames = parse_frames(args.frames, count)
    if frames is None:
        spans = [r for t in tracks for r in t['ranges']]
        if not spans:
            raise CliError('No self-contact intervals to render; pass --frames A-B.', EXIT_USAGE)
        frames = max(0, min(a for a, _ in spans) - MARGIN_FRAMES), min(count - 1, max(b for _, b in spans) - 1 + MARGIN_FRAMES)
    wanted = list(range(frames[0], frames[1] + 1, args.step))
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
    obs = videos = None
    if 'camera' in views:
        from .review import load_exmr, read_frames
        session.info('Loading cameras and video frames ...')
        obs = load_exmr(project)
        videos = read_frames(obs.video, set(wanted))
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
    report = dict(version=1, command='render', project=str(project), model=model, result=kind, frames=[frames[0], frames[1]], step=args.step,
                  views=views, layers=[dict(label=l.label, role=l.role, kind=l.kind) for l in layers],
                  geometry='SMPL-X neutral LBS with shape and pose correctives (GUI Mesh view); not the bound display skin',
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
