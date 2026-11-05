"""Original vs corrected display: translucent original body, self-contact lines and discs, 2D keypoints. Display only.

The original layer is the pre-self-contact pose when the corrected result holds self-contacts, otherwise the uncorrected
base result. It is drawn with the model's SMPL-X mesh path, never the bound display skin.
"""
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path
from threading import Condition

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QThread, Qt, Signal
from PySide6.QtGui import QColor, QPen

PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
GHOST_COLOR = (.80, .83, .90)
COCO_BODY = ((5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16))
WORKER_SECONDS = 300


def residual_color(value):
    if value is None:
        return QColor('#9a9a9a')
    return QColor('#6cc070') if value <= .02 else QColor('#e0b050') if value <= .04 else QColor('#e06a5a')


def cache_folder(project, kind):
    from .paths import HOME
    key = hashlib.sha1(str(Path(project).resolve()).lower().encode('utf-8')).hexdigest()[:16]
    return HOME / 'cache' / 'selfcontact_compare' / key / kind


def cached_record(project, kind, folder):
    from .compare_worker import SCHEMA, sources
    try:
        record = json.loads((folder / 'compare.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None
    if record.get('schema') != SCHEMA or record.get('sources') != sources(project, kind):
        return None
    if not all((folder / record[key]).is_file() for key in ('mesh', 'joints', 'keypoints') if key in record):
        return None
    return record


def run_worker(project, kind, folder, cancelled):
    """Torch and SMPL-X stay in the shared runtime; the editor and its loader thread only read the written arrays."""
    from .paths import worker_command
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    process = subprocess.Popen(worker_command('selfcontact_compare', '--project', str(project), '--kind', kind, '--out', str(folder)),
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=environment,
                               creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    deadline = time.monotonic() + WORKER_SECONDS
    try:
        while True:
            if cancelled():
                raise InterruptedError()
            if time.monotonic() > deadline:
                raise ValueError('Preparing the original layer timed out.')
            try:
                _, error = process.communicate(timeout=.1)
                break
            except subprocess.TimeoutExpired:
                pass
        if process.returncode:
            lines = error.decode('utf-8', 'replace').strip().splitlines()
            raise ValueError(lines[-1] if lines else 'Comparison worker failed.')
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()


def load(project, kind, cancelled=lambda: False):
    """Original layer, per-frame max joint delta and optional keypoints for one corrected result; None if not corrected."""
    from .motion_tools import base_result, result_is_current
    from .result_artifacts import file_stat
    from .preview_data import FrameArray
    project = Path(project).resolve()
    base = base_result(kind)
    if kind == base or not (project / f'{kind}_joints.npz').is_file():
        return None
    with np.load(project / f'{kind}_joints.npz', allow_pickle=False) as data:
        corrected = np.asarray(data['joints'], np.float32)
    report_path = project / f'{kind}_correction.json'
    report = json.loads(report_path.read_text(encoding='utf-8')) if report_path.is_file() else {}
    contact = report.get('self_contact') or {}
    before = project / f'{kind}_before_selfcontact.pt'
    has_before = before.is_file() and contact.get('before_stat') == file_stat(before)
    folder = cache_folder(project, kind)
    record = None
    if has_before or (project / 'exmr_keypoints.pt').is_file():
        record = cached_record(project, kind, folder)
        if record is None:
            run_worker(project, kind, folder, cancelled)
            record = cached_record(project, kind, folder)
            if record is None:
                raise ValueError('The motion changed while the original layer was prepared.')
    if cancelled():
        raise InterruptedError()
    result = dict(project=project, kind=kind, base=base, mesh=None, keypoints=None, residuals=contact.get('tracks') or [],
                  warning='')
    if record and record.get('joints') and has_before:
        result.update(source='before', label='before self-contacts', joints=np.load(folder / record['joints']),
                      mesh=FrameArray(folder / record['mesh'], max_blocks=4))
    else:
        path = project / f'{base}_joints.npz'
        if not path.is_file() or not result_is_current(project, base, ignore_contacts=True):
            return dict(result, source=None, joints=None, delta=None, label='', warning='Original motion preview unavailable.')
        with np.load(path, allow_pickle=False) as data:
            result.update(source='base', label='uncorrected ' + base, joints=np.asarray(data['joints'], np.float32))
        if has_before is False and before.is_file():
            result['warning'] = 'Pre-self-contact pose is outdated; showing the uncorrected motion. Apply Correction to refresh it.'
    ghost = result['joints']
    if ghost.shape != corrected.shape:
        return dict(result, source=None, joints=None, delta=None, warning='Original and corrected motions differ in length.')
    result['delta'] = np.linalg.norm(corrected - ghost, axis=-1).max(axis=-1).astype(np.float32)
    if record and record.get('keypoints'):
        result['keypoints'] = np.load(folder / record['keypoints'])
    return result


class CompareLoader(QThread):
    ready = Signal(int, object)
    failed = Signal(int, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.condition = Condition()
        self.generation = 0
        self.pending = None
        self.stopping = False

    def request(self, project, kind):
        with self.condition:
            self.generation += 1
            self.pending = (self.generation, project, kind)
            self.condition.notify()
            return self.generation

    def cancel(self):
        with self.condition:
            self.generation += 1
            self.pending = None

    def close(self):
        with self.condition:
            self.stopping = True
            self.condition.notify()
        self.wait()

    def run(self):
        while True:
            with self.condition:
                while self.pending is None and not self.stopping:
                    self.condition.wait()
                if self.stopping:
                    return
                token, project, kind = self.pending
                self.pending = None
            cancelled = lambda: self.stopping or token != self.generation
            try:
                value = load(project, kind, cancelled)
                if not cancelled():
                    self.ready.emit(token, value)
            except InterruptedError:
                pass
            except Exception as error:
                if not cancelled():
                    self.failed.emit(token, str(error))


class GhostRenderer:
    """Translucent SMPL-X surface sharing the main mesh renderer's model buffers and skinning shaders."""

    FRAGMENT = """#version 430 core
in vec3 surfaceNormal;
out vec4 color;
uniform vec3 bodyColor;
uniform float alpha;
void main() {
    vec3 n=normalize(surfaceNormal);
    float key=max(0,dot(n,normalize(vec3(-.4,.8,-.6))));
    color=vec4(bodyColor*(.45+.5*key),alpha);
}
"""

    def __init__(self, shared):
        from .mesh_gl import MeshRenderer, VERTEX
        from PySide6.QtOpenGL import QOpenGLShader
        self.mesh = MeshRenderer(shared)
        self.program = MeshRenderer.program(((QOpenGLShader.Vertex, VERTEX), (QOpenGLShader.Fragment, self.FRAGMENT)))

    def draw(self, eye, basis, width, height, alpha):
        from PySide6.QtGui import QMatrix4x4, QVector3D
        from shiboken6 import VoidPtr
        mesh, gl = self.mesh, self.mesh.gl
        projection = QMatrix4x4()
        projection.perspective(40, width / max(1, height), .03, 2000)
        view = np.eye(4)
        view[:3, :3] = basis * np.array([[1], [1], [-1]])
        view[:3, 3] = -view[:3, :3] @ eye
        self.program.bind()
        self.program.setUniformValue('mvp', projection * QMatrix4x4(*view.reshape(-1).tolist()))
        self.program.setUniformValue('sceneTransform', QMatrix4x4(*np.eye(4).reshape(-1).tolist()))
        self.program.setUniformValue('bodyColor', QVector3D(*GHOST_COLOR))
        # PySide resolves setUniformValue(location, float) to the int overload: glUniform1i on a float is GL_INVALID_OPERATION.
        gl.glUniform1f(self.program.uniformLocation('alpha'), float(alpha))
        gl.glEnable(0x0B71)
        gl.glDisable(0x0B44)
        # Offset behind coincident corrected surfaces: untouched body parts stay hidden inside the corrected body.
        gl.glEnable(0x8037)
        gl.glPolygonOffset(1., 4.)
        mesh.vao.bind()
        # Depth-only pass, then one blended pass of the front-most original surface (no stacked self-overlap).
        gl.glColorMask(False, False, False, False)
        gl.glDepthMask(True)
        gl.glDepthFunc(0x0201)
        gl.glDrawElements(0x4, mesh.index_count, 0x1405, VoidPtr(0))
        gl.glColorMask(True, True, True, True)
        gl.glDepthMask(False)
        gl.glDepthFunc(0x0203)
        gl.glEnable(0x0BE2)
        # Keep destination alpha: the widget framebuffer is composited by Qt, and alpha < 1 would show through it.
        gl.glBlendFuncSeparate(0x0302, 0x0303, 0, 1)
        gl.glDrawElements(0x4, mesh.index_count, 0x1405, VoidPtr(0))
        mesh.vao.release()
        gl.glDisable(0x0BE2)
        gl.glDepthMask(True)
        gl.glDepthFunc(0x0201)
        gl.glDisable(0x8037)
        gl.glDisable(0x0B71)
        self.program.release()
        mesh.check('original layer')

    def close(self):
        self.mesh.close()
        self.program.removeAllShaders()


class CompareOverlay:
    def __init__(self, window):
        self.w = window
        self.show = False
        self.opacity = .35
        self.data = None
        self.message = ''
        self.residual_lookup = {}
        self.applied = False
        self.key = None
        self.token = 0
        self.renderer = None
        self.loader = CompareLoader(window)
        self.loader.ready.connect(self.loaded)
        self.loader.failed.connect(self.load_failed)
        self.loader.start()

    # Data -------------------------------------------------------------------------------------------------------------

    def request(self, project, kind, preview):
        key = (str(project), kind, id(preview))
        if key == self.key:
            return
        self.key = key
        self.clear_data()
        if project is None or not kind.startswith('corrected_'):
            self.loader.cancel()
            self.publish()
            return
        self.message = 'Preparing original layer...'
        self.token = self.loader.request(project, kind)
        self.publish()

    def clear(self):
        self.key = None
        self.loader.cancel()
        self.clear_data()
        self.publish()

    def clear_data(self):
        if self.data is not None and self.data.get('mesh') is not None:
            self.data['mesh'].close()
        self.data = None
        self.message = ''
        self.residual_lookup = {}

    def loaded(self, token, data):
        if token != self.token:
            if data and data.get('mesh') is not None:
                data['mesh'].close()
            return
        self.clear_data()
        self.data = data
        self.message = (data or {}).get('warning', '')
        self.residual_lookup = {}
        for row in (data or {}).get('residuals', []):
            if isinstance(row, dict) and 'start' in row:
                self.residual_lookup.setdefault(row.get('id'), []).append((int(row['start']), int(row['end']), row.get('residual_max_m')))
        self.publish()

    def load_failed(self, token, message):
        if token == self.token:
            self.clear_data()
            self.message = 'Original layer unavailable: ' + message
            self.publish()

    def publish(self):
        data = self.data
        delta = data.get('delta') if data else None
        self.w.slider.set_compare_delta(delta, ('vs ' + data['label']) if delta is not None else '')
        controller = getattr(self.w, 'self_contacts', None)
        if controller is not None:
            controller.sync()
        self.w.motion_view.update()
        if hasattr(self.w, 'camera_tools'):
            self.w.camera_tools.canvas.mirror()

    def available(self):
        return bool(self.data and self.data.get('joints') is not None)

    def ghost_mesh(self):
        if self.data.get('mesh') is not None:
            return self.data['mesh']
        preview = self.w.preview_data
        return preview.arrays.get(self.data['base'] + '_mesh') if preview is not None else None

    def close(self):
        self.loader.close()
        self.clear_data()

    # 3D view ----------------------------------------------------------------------------------------------------------

    def release_renderer(self, view):
        if self.renderer is not None and view.isValid():
            view.makeCurrent()
            self.renderer.close()
            view.doneCurrent()
        self.renderer = None

    def draw_ghost_mesh(self, view, painter, index):
        from .viewport import OPENGL_VIEW
        from .mesh_gl import MeshRenderer
        owner = view.mesh_renderer
        mesh = self.ghost_mesh()
        if (not OPENGL_VIEW or mesh is None or view.mesh_asset is None or type(owner) is not MeshRenderer or owner.asset_key is None
                or index >= len(mesh) or view.mesh_presented_index != index):
            return False
        painter.beginNativePainting()
        try:
            if self.renderer is not None and self.renderer.mesh.shared is not owner:
                self.renderer.close()
                self.renderer = None
            if self.renderer is None:
                self.renderer = GhostRenderer(owner)
            self.renderer.mesh.load(view.mesh_asset)
            self.renderer.mesh.deform(mesh[index], ('ghost', id(mesh), index))
            self.renderer.draw(*view._paint_camera, view.width(), view.height(), self.opacity)
            return True
        except Exception as error:
            self.message = 'Cannot draw the original mesh: ' + str(error)
            return False
        finally:
            painter.endNativePainting()

    def paint_view(self, view, painter, index, lines3d):
        """Called by the viewport after the corrected mesh, inside its depth buffer."""
        data = self.data
        if self.show and self.available() and index < len(data['joints']):
            self.draw_ghost_mesh(view, painter, index)
            ghost = data['joints'][index]
            color = QColor(205, 208, 214, int(90 + 140 * self.opacity))
            lines3d(np.stack([ghost[[p for p in PARENTS[1:]]], ghost[1:]], axis=1), color, 1.6)
            painter.setPen(QColor('#c3c7cd'))
            painter.drawText(QRectF(18, 81, view.width() - 36, 20), Qt.AlignLeft, 'Original (grey): ' + data['label'])
        elif self.show and self.message:
            painter.setPen(QColor('#e0b06c'))
            painter.drawText(QRectF(18, 81, view.width() - 36, 20), Qt.AlignLeft, self.message)
        if view.show_foot_locks and view.joints is not None:
            self.paint_contacts(view, painter, index, lines3d)

    def contact_points(self, view, index, track):
        """Palm of the effector and the target point on the corrected body (the other palm for hand targets)."""
        from .mixamo_rig import BODY_NAMES
        joints = view.joints[min(index, len(view.joints) - 1)]
        rig = view.target_frames[index] if view.target_frames is not None and index < len(view.target_frames) else None

        def palm(hand):
            if rig is not None and rig.shape == (52, 4, 4):
                points = rig[:, :3, 3]
                return (points[20 + hand] + points[np.array([25, 28, 31, 34]) + 15 * hand].mean(axis=0)) * .5
            return np.asarray(joints[20 + hand], float)
        effector = palm(0 if track['effector'] == 'LeftHand' else 1)
        target = track['target']
        point = palm(0 if target == 'LeftHand' else 1) if target in ('LeftHand', 'RightHand') else np.asarray(joints[BODY_NAMES.index(target)], float)
        return effector, point

    def residual(self, track, index):
        for start, end, value in self.residual_lookup.get(track['id'], []):
            if start <= index < end:
                return value
        return None

    def paint_contacts(self, view, painter, index, lines3d):
        from .selfcontact_rows import ROW_COLORS, row_label
        slider = self.w.slider
        if not slider.sc_visible:
            return
        for i, track in enumerate(slider.sc_rows.tracks):
            if not track['enabled'] or not any(s <= index < e for s, e in track['ranges']):
                continue
            effector, target = self.contact_points(view, index, track)
            if not (np.isfinite(effector).all() and np.isfinite(target).all()):
                continue
            value = self.residual(track, index) if self.applied else None
            color = residual_color(value)
            lines3d([[effector, target]], color, 2.2)
            xy, depth, focal = view.project(np.stack([effector, target]))
            row_color = QColor(ROW_COLORS[i % len(ROW_COLORS)])
            fill = QColor(row_color)
            fill.setAlpha(110)
            for point, z in zip(xy, depth):
                if z <= .08:
                    continue
                radius = max(4., min(30., focal * .045 / z))
                painter.setBrush(fill)
                painter.setPen(QPen(row_color, 1.5))
                painter.drawEllipse(QPointF(*point), radius, radius)
            if min(depth) > .08:
                distance = float(np.linalg.norm(effector - target))
                text = row_label(track) + (f'  residual {value * 100:.1f} cm' if value is not None else
                                           '  pending: Apply Correction' if not self.applied else '') + f'  ({distance * 100:.0f} cm joint gap)'
                middle = xy.mean(axis=0)
                painter.setPen(color)
                painter.drawText(QPointF(middle[0] + 8, middle[1] - 6), text)
        painter.setBrush(Qt.NoBrush)

    # Camera preview ---------------------------------------------------------------------------------------------------

    def camera_actors(self, view):
        if self.show and self.available() and view.single_actor_visible and 0 <= view.index < len(self.data['joints']):
            yield np.asarray(self.data['joints'][view.index], float), np.array(PARENTS), '#b8bcc4'

    def paint_keypoints(self, painter, rect, camera, index, canvas):
        """2D detections over the video: ExMR COCO-63 from the cache, or the saved COCO-17 on the camera canvas only."""
        if index is None:
            return
        width, height = camera['size']
        sx, sy = rect.width() / width, rect.height() / height
        points = confidence = None
        if self.data is not None and self.data.get('keypoints') is not None and index < len(self.data['keypoints']):
            values = self.data['keypoints'][index]
            points, confidence = values[:, :2], values[:, 2]
        elif canvas and self.w.preview_data is not None:
            arrays = self.w.preview_data.arrays
            if arrays.get('keypoints') is not None and index < len(arrays['keypoints']):
                points = np.asarray(arrays['keypoints'][index], float)
                confidence = np.asarray(arrays['confidence'][index], float).reshape(-1)
        if points is None:
            return
        mapped = np.column_stack((rect.left() + points[:, 0] * sx, rect.top() + points[:, 1] * sy))
        painter.setPen(QPen(QColor(124, 236, 255, 200), 2))
        for a, b in COCO_BODY:
            if min(confidence[a], confidence[b]) >= .3:
                painter.drawLine(QPointF(*mapped[a]), QPointF(*mapped[b]))
        painter.setPen(Qt.NoPen)
        for j, (point, score) in enumerate(zip(mapped, confidence)):
            if score >= .3 and math.isfinite(point[0]) and math.isfinite(point[1]):
                painter.setBrush(QColor('#fff6cf') if j < 23 else QColor('#9fe8a8'))
                size = 3. if j < 23 else 1.8
                painter.drawEllipse(QPointF(*point), size, size)
        painter.setBrush(Qt.NoBrush)
