# RGB MoCap command line (`rgbmocap-cli`)

`rgbmocap-cli.cmd` runs the same processing as the GUI without opening a window. It builds the job lists the GUI builds for the
same settings (including the ExMR/FlowHMR stage joining and the multi-person batching) and starts the same worker processes, so
projects created on the command line open normally in the app and vice versa.

```
rgbmocap-cli.cmd [global options] COMMAND [options]
rgbmocap-cli.cmd COMMAND --help
```

Run it from any folder; relative paths are resolved against your current folder. Install first with `install.cmd`.

## Global options

These work before or after the command.

| Option | Effect |
|---|---|
| `--dry-run` | Print the worker jobs (and batch request bodies) that would run, then exit 0. Needs no models. Jobs whose project is only known after an earlier stage show placeholders: `<project>` (from `prepare`), `<take:ID>` (from each `person` job), `<request>`, `<frames>`. |
| `--json` | Print a machine-readable summary on stdout at the end; progress goes to stderr. |
| `--quiet` | Print only results and errors. |
| `--low-memory` | Low Memory resource mode (the GUI's Resource Mode). |
| `--no-resource-safety` | Turn Resource Safety off for this run only (the saved preference is not changed). |
| `--no-lock` | Do not take the app lock. Normally the CLI refuses to run while RGB MoCap, an install or an update is running. |
| `--view` | When the command succeeds, open its project in the RGB MoCap GUI (its own window and console). If the GUI is already open, the CLI prints the folder to open with **File > Open Project...** instead. `batch` opens the last clip that succeeded. |

`--help`, `--dry-run`, `models`, `review` and `fix` never take the lock, so they work while the app is open (`review` and `fix` only read projects and write to their own output folders). `contacts list`, `contacts feet list`, `contacts place list`, `contacts smooth list` and `contacts suggest` without `--accept` do not take it either.

Exit codes: `0` ok, `1` failed, `2` invalid arguments, `3` missing models, `4` the app (or an install/update) is running,
`130` cancelled with Ctrl+C. Ctrl+C stops the running worker and its child processes; completed stages are reused next time.

## Models

```
rgbmocap-cli.cmd models [--model exmr|flowhmr|gvhmr] [--people] [pipeline options]
```

Lists the model files the selected pipeline needs and which are missing (all three models when `--model` is omitted); exit 3 if
anything is missing. `--people` adds person detection (YOLO, OSNet). FlowHMR with `--flow-camera static` does not need VGGT;
`--hamer` adds the HaMeR files. Install models with the app's Model Manager.

Every processing command checks its models before starting any work, exactly like Reconstruct 3D, and exits with code 3 listing
the missing files.

## Capture a video

```
rgbmocap-cli.cmd capture VIDEO [--model exmr|flowhmr|gvhmr] [--start S] [--end S] [--people all|1,3] [--out PATH] [options]
```

Without `--people` this is the GUI's Reconstruct 3D on a freshly opened video: `prepare`, the model's stages, correction and
export. The default model is ExMR, as in the app. Start and end are seconds on the source video (default: the whole video). The
model project folder (`projects/<video>_<key>_project/<model>`) is printed at the end; reuse it with the other commands.

With `--people`, people are found first (Find person), then the selected people are reconstructed (`all`, or the ids printed in
the people table). The selection is saved as the checked people in `people.json`, as checking them in the app would.

`--out` exports the result when processing has finished (see Export). A path ending in `.fbx`, `.bvh` or `.blend` is a file;
anything else is a folder that receives `<person name>.<format>` (or one file per take for several people).

Default flows (what `--dry-run` prints):

| Model | Stages | Result |
|---|---|---|
| ExMR | prepare, track, exmr_batch (preprocess + features + ExMR), correct, export | `corrected_initial` |
| FlowHMR | prepare, track, flow_camera, flow_batch (features + FlowHMR), correct, flow_export | `corrected_initial` |
| GVHMR | prepare, track, gvhmr, pva, refine, footmr, handflow, correct, export | `corrected_handflow_footmr_refined` |

### Pipeline options

Options default to the app's defaults for a new video; for an existing project (`reconstruct`, `correct`) they default to the
project's saved settings. Every on/off option also has a `--no-` form.

| Option | Applies to | Meaning |
|---|---|---|
| `--refine`, `--footmr`, `--handflow` | GVHMR | HTD refinement, FootMR, HandFlow fingers (default on) |
| `--correction` / `--no-correction` | all | Motion (contact) correction (default on) |
| `--camera static\|moving` | ExMR | Static camera (default) or moving camera / SimpleVO |
| `--smoothing` / `--no-smoothing` | ExMR | Smooth Motion (default on) |
| `--flow-camera auto\|static\|N` | FlowHMR | Camera sampling: auto (default), static, or every N frames (1, 2, 5, 10, 30) |
| `--fast-sam`, `--sam-batch4`, `--sam-gpu-crop`, `--hand-stride4` | FlowHMR | Acceleration options (default on) |
| `--hamer` | FlowHMR | HaMeR finger refinement (default off) |
| `--preset weak\|normal\|strong` | correction | Lock preset (default normal) |
| `--floor-mode flat\|multi_level` | correction | Flat ground (default) or multi level |
| `--floor M\|auto` | correction | Manual floor height in meters (world Y), or automatic (default) |
| `--pelvis-offset M` | correction | Pelvis height offset, -0.25 to 0.25 m |
| `--foot-lock`, `--hand-lock` | correction | Contact locks (foot on, hand off by default) |

## Find people

```
rgbmocap-cli.cmd people VIDEO [--model ...] [--start S] [--end S]
rgbmocap-cli.cmd people --project P
```

Runs Find person and prints a table of ids, names, checked state, first/last frames, observed frames and takes (`--json` lists
them). The printed people root is the project to pass to `reconstruct`, `correct` and `export`. Person detection needs an NVIDIA
CUDA GPU.

## Reconstruct an existing project

```
rgbmocap-cli.cmd reconstruct --project P [--people all|checked|1,3] [--start S] [--end S] [pipeline options] [--out PATH]
```

`--project` accepts a project folder (the workspace root containing the model folders; `--model` selects one), a model project, a
people root or a person take. Single-person projects are re-run like Reconstruct 3D with the loaded project. People roots
reconstruct the selected people (default: the checked people) as one `person` job each; the takes they announce are then
processed together: ExMR and GVHMR stages are combined into batch jobs, and correction/export run last so the shared scene floor
is resolved once. If the workspace has no project for `--model` yet, the workspace video is processed as a new capture.

## Apply correction

```
rgbmocap-cli.cmd correct --project P [--preset ...] [--floor ...] [--floor-mode ...] [--pelvis-offset M] [--foot-lock ...] [--hand-lock ...]
                         [--people 1,3] [--result KIND] [--grounding vertical|camera] [--no-character-refresh]
```

The GUI's Apply Correction: the correction settings are written to the project, then `correct` and `export` run. On a people root
the shared floor (`scene_floor.json`) is updated and every take with an available base result is corrected again; the lock settings
are written to the takes of `--people` (default: all corrected takes). Characters retargeted from a corrected take are marked as
needing retargeting and re-retargeted afterwards, as in the app; `--no-character-refresh` only marks them.

The flat floor model shifts the root so each support frame lands on one floor height. When that grounding shift spans
more than 0.15 m (the source world's floor drifts or tilts), `<kind>_correction.json` `warning` says so: the body may
move off the video person (review `body_offset`); Multi Level keeps the source elevation. For a moving ExMR camera, the
GUI camera gizmos and the Blender export camera follow the root change correction made, so the body keeps its place in
view. `--grounding camera` (single static-camera ExMR take, Flat Ground; saved with the settings, the GUI does not show it)
also moves the root toward or away from the GUI's fixed camera, smoothed over 2 s, so the drop keeps the body on the
person; `<kind>_correction.json` `camera_grounding` gives the depth change. Default `vertical`.

## Export

```
rgbmocap-cli.cmd export --project P --out PATH [--format fbx|bvh|blend] [--people all|1,3] [--result KIND]
                        [--bvh-profile generic|biped] [--tpose] [--overwrite]
                        [--no-camera-animation] [--camera-scale X] [--refine-camera-rotation]
```

Exports the committed source motion through the same save workers as the app. The result defaults to the project's selection.
If the motion is missing, out of date or has a pending correction, the export stops and asks you to run `correct` or
`reconstruct` first.

- One take: `--out` is a file, or a folder receiving `<person name>.<format>`. Existing files are kept unless `--overwrite`.
- Several people or takes (people roots, default: checked people): `--out` is a folder; files are named
  `<name>[_take_NN].<format>` and, as in the app, never replace existing files (a `_2` suffix is added) unless `--overwrite`.
- People takes are exported in the shared scene placement, as in the app.
- Blender (`.blend`) creates a new folder `<name>/<name>.blend` with the video; it supports one take at a time. The camera options
  apply to Blender export only.

## Batch

```
rgbmocap-cli.cmd batch DIR --out OUTDIR [--model ...] [--format fbx|bvh|blend] [pipeline options] [--continue-on-error] [--overwrite]
```

Captures every `.mp4`, `.mov`, `.mkv`, `.avi` and `.webm` in DIR (not subfolders) as a single person and exports each result to
`OUTDIR/<clip name>.<format>`. Models and output conflicts are checked before the first clip. A summary table is printed at the end;
the exit code is non-zero if any clip failed. Without `--continue-on-error` the batch stops at the first failure.

## Characters

Import humanoid characters into a project, bake a reconstructed person's motion onto them and export the result,
exactly as **File > Import Character...**, **Retarget** and **Export Character...** do in the GUI. Characters are stored
in `<project>\characters\` (shared by all models of the project); up to 8 per project. Retargeting and export use the
bundled Blender in `runtime_character` and run no AI models.

`--project` accepts the project folder, a model folder inside it or a person take. `CHAR` is a character id, a unique
id prefix or its name (`character list` shows them).

| Command | What it does |
| --- | --- |
| `character import --project P FILE` | Import a Mixamo-rig `.fbx`, `.vrm` or `.pmx` file. |
| `character list --project P` | List characters, their kind, visibility and assigned motion. |
| `character sources --project P [--result KIND]` | List people whose motion can be retargeted (exit 1 if none). |
| `character retarget --project P CHAR --person NAME\|ID [--result KIND]` | Bake that person's takes onto the character. |
| `character export --project P CHAR --out PATH [--format fbx\|vmd] [--no-mesh] [--overwrite]` | Export the baked motion. |
| `character remove --project P CHAR` | Remove the character from the project list. Its files are kept on disk. |

- `--result` defaults to the project's selected result. A corrected result must be current and exported first
  (run `correct`), otherwise retargeting is refused.
- Export: one take writes one file (`--out hero.fbx`, or a folder to get `<name>.fbx`); several takes write
  `<name>_take_1.fbx`, `<name>_take_2.fbx`, ... into the `--out` folder. Existing files are only replaced with
  `--overwrite`. `--no-mesh` saves skeleton and animation only. `--format vmd` (PMX characters only) saves body and
  finger motion at 30 fps for MMD.
- `correct` also re-retargets characters baked from the corrected takes. Until that succeeds they are marked
  **needs retarget** and cannot be exported; run `character retarget` again after any other source change.
- `--dry-run` prints the character worker request as JSON without running it; `--json` prints a summary object
  (`{"ok": true, ...}` or `{"ok": false, "error": ...}`).

## Fix animation (IK pins)

```
rgbmocap-cli.cmd fix INPUT.fbx --spec fixes.json --out OUTPUT.fbx [--report report.json] [--overwrite]
```

Applies corrections from a JSON spec to an FBX animation with the bundled Blender (`runtime_character`), headless.
Pins hold a hand or foot on another bone (a hand grabbing a foot) or on a fixed point (a planted foot, a hand on a
table); smoothing calms rotation jitter. The output keeps the input's armature, bone names (including the `*_end`
leaf bones), rest pose, mesh, frame range and fps; only the pinned and smoothed frames change, every other frame is
the input animation (round trip within 3e-6 m).

- `--spec -` reads the spec from stdin. `--out` and `--report` are only replaced with `--overwrite`.
- `--dry-run` prints the validated spec and the resolved plan (bones, IK chains, changed frames) without starting
  Blender. `--json` prints the report as the summary.
- Spec errors (malformed JSON, unknown keys or bones, frames outside the clip, overlapping pins) exit with code 2.

```json
{
  "version": 1,
  "pins": [
    {"effector": "RightHand", "target": "RightFoot", "frames": [381, 421], "ease": 6, "distance": 0.08},
    {"effector": "LeftFoot", "target": "world", "frames": [100, 119], "ease": 5}
  ],
  "smooth": [{"bones": ["RightHand", "RightForeArm"], "frames": [100, 140], "window": 5}]
}
```

| Pin key | Meaning |
| --- | --- |
| `effector` | Bone to move: `LeftHand`, `RightHand`, `LeftFoot`, `RightFoot`, `LeftToeBase` or `RightToeBase`. A `prefix:` namespace is ignored (`mixamorig:RightHand`). |
| `target` | Bone to stick to (any bone outside the pinned limb), or `world`: the effector's mean position over the range. |
| `frames` | `[first, last]`, inclusive. Frame 1 is the file's first key, as Blender numbers frames after a default FBX import (the app's FBX files run 1..N). |
| `ease` | Frames to blend in before `first` and out after `last` (smoothstep). Frames `first - ease` and earlier, `last + ease` and later are untouched. |
| `distance` | Optional. Effector-to-target distance in meters (e.g. palm thickness), along the direction (in the target bone's space) at the frame in the range where the two are closest. Omitted: that frame's original offset is held. Bone targets only. |
| `offset` | Optional `[x, y, z]` meters, overrides `distance`: the effector position in the target bone's local space. For `world`, added to the held point (armature space; Y up in the app's FBX files). |
| `chain` | Optional. `2` (default): upper arm + forearm, or thigh + shin. `3` (arms only) also turns the shoulder (clavicle), only when the arm cannot reach and by at most 30 degrees. |
| `rotation` | Optional. `keep` (default): the effector keeps its original world rotation. `follow`: it keeps its rotation relative to the target from the closest frame (`world`: holds that frame's rotation). |

`smooth` entries: `bones` (list), `frames` (`[first, last]`) and `window` (at least 2). The bones' local rotations
are Gaussian-smoothed over `window` frames; the effect ramps in and out over the first and last `window` frames of
the range and frames outside it are unchanged. Smoothing runs before the pins.

How pins are solved:

- Pins run in spec order, each on the result of the ones before it, so a second pin can move the target of the
  first. Pins on the same limb must not overlap, ease frames included.
- Each frame is solved analytically. The elbow or knee stays on the side it bends to in the original motion (the
  pole is the original joint pushed outward along its bend) and the limb keeps its twist, so it cannot flip. Only
  the chain bones and the hand or foot get new rotation keys, only on the changed frames.
- An unreachable target straightens the limb toward it; the report lists those frames and the remaining distance.
  If a hand cannot reach a foot (arm's length is shorter than shoulder to foot), add a second pin moving the foot
  to the hand so the two meet.

The report (printed; all fields with `--report` or `--json`) has, per pin: the effector-target distance before and
after at the range start, middle and end plus its min and max, `max_error` (effector to its wanted point),
`reach_failures` (frames, largest residual), `mid_joint_max_shift` (how far the elbow or knee moved),
`frames_changed`, and the offset or world point used; per smooth range, the largest rotation change per bone.

Limitations: binary FBX with one animated armature and quaternion bone keys (Blender's FBX import); the output has
no bind-pose table unless it has a skinned mesh (the rest pose is kept in the bone defaults); only rotations are
changed, never bone translations or the root motion.

## Review a motion

```
rgbmocap-cli.cmd review --project P [--model exmr] [--result KIND] [--motion EDITED.fbx] [--out DIR] [--frames A-B] [--max-images N]
                        [--compare original|before-selfcontact] [--no-anatomy]
```

Measures how well a motion matches the source video and flags problems, many with a ready `fix` pin or smoothing
entry. It reads the project's 2D keypoints, cameras and video and runs no AI models. It writes into `--out`
(default `<model project>\review\<result or FBX name>-<time>\`):

- `summary.md`: ranked issues with frames, numbers, image names and the suggested fix per issue, plus global stats.
- `report.json`: everything machine-readable: issues, keypoint swaps, stats, thresholds, per-frame series and
  `suggested_fixes`, a valid `fix` spec (also saved as `suggested_fix.json`).
- `images\*.jpg`: the video frame with confident 2D keypoints (green) and the 3D motion projected with that frame's
  camera after the 2D alignment (magenta), the joints involved and the metric values. There is one image per issue's worst
  frame, then start/end frames, a few evenly spaced samples and `contact_sheet.jpg`, up to `--max-images` (default 24).

`--motion` reviews an FBX (for example a `fix` output) against the same video and keypoints, so before and after can be
compared. Its bone positions are read with the bundled Blender (bones matched by name, `prefix:` ignored; a missing body
bone is an error). `--result` picks the project result (default: the project selection). Floor and contacts come from
that result's correction files when present. Report frames and `--frames` are video frames (0-based); Blender frames
and fix frames are video frame + 1. Only ExMR projects are supported for now.

Method: the camera follows the project's ExMR camera mode (`exmr_camera.json`; `report.json` `camera` names the one
used). A static camera is one fixed camera, the GUI camera view's: the frame-0 pairing of the original in-camera and
world roots. So ExMR's world drift (up to 0.25 m) shows as placement offset in the original result, and Motion Correction
removing it does not. A moving camera is paired per frame with the original roots, and for a corrected result its
centres move by the root change correction made (as the Blender export camera), so a corrected result keeps the
original's place in view. Projects without a saved camera mode keep the per-frame pairing. For pose checks, the
per-frame 2D offset (median of the confident shoulder keypoints, hips weighted low) is removed first and reported
separately as placement drift (aligned pose errors therefore barely depend on the camera mode). Errors
are divided by the torso length in pixels (shoulder middle to hip middle), so thresholds do not depend on resolution.
Keypoints count when their score is at least 0.5. Frames where the 2D detector swapped left and right legs or arms
are relabelled and listed under `keypoint_swaps`.

| Check | Flags (runs of at least 3 frames, gaps of up to 2 bridged) | Suggested fix |
| --- | --- | --- |
| `missed_contact` | A wrist or hand is within 0.25 torso of a foot, the other hand, the head (0.25), a knee (0.20) or a hip (0.15) in 2D, but more than 0.15 m away in 3D (head 0.25 m, hip 0.22 m). | Hand pin onto the nearest bone. If the arm is too short to reach, `chain` 3 plus a second pin moving the target limb to the hand. |
| `pose_error` | An arm (elbow, wrist) or leg (knee, ankle) differs from the keypoints by more than 0.35 torso after alignment. | None (re-run reconstruction or edit by hand). |
| `foot_slide` | During ground contact, the slower of ankle and toe moves horizontally faster than 0.15 m/s. | `world` pin on that foot. |
| `floor_penetration` | The ankle or toe joint is more than 1 cm below its resting height above the floor during contact. | None (`correct --pelvis-offset` / `--floor`). |
| `jitter` | A joint's acceleration (pelvis-relative; the pelvis absolute) exceeds 8 times its median and 20 m/s^2, and 3 times its local median. | `smooth` on the limb's bones. |
| `placement_drift` | The whole body sits more than 0.25 torso off the person in the video (informational). | None. |
| `body_offset` | Corrected results: for at least 1 s the body sits more than 0.2 torso farther off the person than the uncorrected base result does, each through its own camera (medium from 0.3, high from 0.4 median excess). Metrics give both offsets, the direction (below / above / sideways) and `mode`, `max_root_y_offset_m` and `warning` from `<kind>_correction.json`. | None (check the GUI camera view; `correct --floor-mode multi_level` or `--floor`). |
| `hyperextension` | An elbow or knee bends more than 5 deg the wrong way about its anatomical hinge (high from 25 deg). | None. |
| `locked_limb` | An elbow is held at 170 deg interior or straighter for 6+ frames (knee 178); high when locked at 177+ for 10+ frames, the typical IK reach failure. | None (move the target limb / reciprocal). |
| `sudden_fold` | An elbow or knee dips more than 35 deg below the angles around it within 0.35 s (an easing fold; knees one level lower). | None. |
| `wrist_limit` | The wrist leaves its range relative to the forearm (flexion 80, extension 70, radial 25, ulnar 40 deg). | None (grip). |
| `joint_twist` | A knee (40 deg), elbow (80), wrist (80), upper arm at the shoulder (90) or thigh at the hip (60) twists about its own bone: candy-wrap risk. | None. |
| `joint_flip` | An arm or leg flips: the upper or lower bone turns more than 45 deg in one frame against the torso, or the elbow / knee swings (while bent below 150 deg) or the hand / foot twists about the lower bone by more than 45 deg in one frame or 120 deg within 5 frames (per 1/30 s; scaled to the clip's fps). High from 150 deg or 1.5 times the step limit. Typically an IK / contact solve changing the elbow branch. | None (re-run `correct`). |
| `finger_pose` | A finger joint is more than 10 deg past its flexion / spread / twist range (SMPL-X hand pose relative to the flat hand). | None. |
| `hand_orientation` | On frames with confident hand keypoints, the 2D wrist-to-middle-knuckle direction differs by more than 50 deg from the motion, or the thumb and pinky sides are swapped. | None. |
| `self_penetration` | On the posed SMPL-X mesh, a hand or forearm is more than 1 cm inside a leg, the head, the torso or the other arm, or the legs pass through each other (high from 2.5 cm). | None (self-contact solver or limb placement). |
| `contact_gap` | During an enabled self-contact interval the hand skin floats more than 1.5 cm from the target skin (high from a 4 cm median). | None (`correct` with the track enabled). |

`suggested_fixes` contains the high and medium issues in rank order. A pin whose limb and ease window would overlap an
earlier pin is left out (noted on the issue), because `fix` rejects overlapping pins. Typical loop: `review`, run `fix`
on the exported FBX with `suggested_fix.json`, then `review --motion fixed.fbx` to confirm the contact is resolved and
no new jitter appeared at the eased frames.

Anatomy checks read joint rotations: for a project result the model's SMPL-X pose (including the 45-D hand pose, relative
to the hand mean), for `--motion` the FBX bone rotations relative to its bind pose (a second Blender read; finger checks
are skipped there because the bind bakes the relaxed hand). Interior angles are 180 deg when straight; the hinge sign
follows the export's knee/elbow axes. `--no-anatomy` skips them. `report.json` lists their thresholds and the confident
hand-keypoint frames under `anatomy`. `self_penetration` and `contact_gap` need the SMPL-X mesh, so they run on project
results only; `anatomy.skin_contact` adds per declared self-contact the penetration depth and skin gap over its frames.

Limb placement: `report.json` `stats.placement` holds each limb's 2D error (median, p90, max; torso units). When the
result has limb placement rows, each row also gets its error before placement, as solved, and in the reviewed motion,
with `kept` false when a later pass (self-contacts, smoothing) pulled the limb away from the video again; `summary.md`
shows them as a table.

`--compare original|before-selfcontact` (corrected results only) also reviews the reference layer over the same frames:
the uncorrected base result, or `<kind>_before_selfcontact.pt` written by `correct` when self-contacts changed the motion
(refused when outdated: the result is stale or the file's stat no longer matches the report).
`report.json` then has a `compare` block: the issue diff (`fixed`, `new`, `unchanged` matched by check, joints and
overlapping frames), untouched-frame identity (against `before-selfcontact`, frames and joints outside the
`self_contact.changed` rows of `<kind>_correction.json` must be bit-identical), the per-frame maximum joint-rotation
change, and `images\compare_delta_<layer>.png` (one column per frame, white ticks = changed ranges).

## Render bodies

```
rgbmocap-cli.cmd render --project P [--model M] [--result KIND] [--frames A-B] [--step N] [--views camera,front,side,orbit]
                        [--compare [original|before-selfcontact]] [--focus BONE,BONE] [--out DIR] [--mp4] [--size PX]
rgbmocap-cli.cmd render --project P --sheet [--frames A-B|A,B,C] [--compare original|before-selfcontact] [--add LABEL=FILE] [--out DIR]
```

Draws the model's own SMPL-X mesh (the GUI Mesh view geometry: shape and pose correctives, never the bound display
skin) with a software rasteriser, so it needs no GPU. Views: `camera` over the video frame with review's camera (ExMR:
one fixed camera for a static camera; per frame for a moving one, following each layer's own correction), `front` and
`side` from the body's mean heading, and `orbit`, four close views around each self-contact
(or `--focus`). `--compare` puts the reference layer (grey) beside the result (orange); without a value it picks
`before-selfcontact` when that file is current (its stat matches `<kind>_correction.json` and the result is not stale),
else `original`, with a note when an outdated file was skipped. A red line marks each self-contact with its palm-to-target
distance. Without `--frames` the self-contact intervals plus 10 frames are rendered. Writes `<view>_f<frame>.png`,
`<view>.mp4` with `--mp4` (FFmpeg) and `render.json` (frames, layers, distances per track and frame, and the largest
vertex difference to the project's GUI mesh preview). Default folder: `<model project>\render\<result>-<time>\`.

`--sheet` writes one comparison image per enabled self-contact track instead (ExMR): one row per frame, columns
**Video** (a crop around the contact), **Before** and **After** drawn over the same crop, then each layer in a close 3D
view of the touching limbs only (90 deg from the camera). Each tile states the hand's depth inside the target or the skin
gap, the arm's depth inside the target, and the angle between the model's wrist-to-knuckle direction (magenta arrow)
and the video's (green). `--frames A-B` (default: the track's ranges) gives four evenly spaced frames, `A,B,C` exactly
those frames. Before is `--compare` (default `original`); `--add LABEL=FILE` adds another result file in the same params
format (for example a saved grip-off result) as a column, repeatable. Writes `sheet_<id>_f<first>-<last>.png` and
`sheet.json` into `--out` (default `<model project>\render\<result>-sheet-<time>\`).

## Self-contacts

```
rgbmocap-cli.cmd contacts list    --project P [--model M]
rgbmocap-cli.cmd contacts add     --project P --effector RightHand --target RightFoot --frames 345-420[,500-510]
                                  [--distance M|auto] [--offset X,Y,Z|auto] [--reciprocal off|auto|always] [--grip off|auto] [--id ID] [--disabled]
rgbmocap-cli.cmd contacts edit    --project P --id ID [--frames ...] [--add-frames ...] [--remove-frames ...] [--accept]
                                  [--target BONE] [--enable|--disable] [--distance ...] [--offset ...] [--reciprocal ...] [--grip ...]
rgbmocap-cli.cmd contacts remove  --project P --id ID | --all
rgbmocap-cli.cmd contacts suggest --project P [--accept] [--min-severity high|medium|low] [--result KIND]
```

A self-contact track holds a hand (`--effector` `LeftHand` or `RightHand`) on another body bone (`--target`, any body
bone except the same arm, for example `RightFoot`, `LeftLeg`, `Head` or the other hand) over frame intervals. `correct`
applies the tracks of the selected result. The tracks are stored in the model project as
`selfcontact_<base>_state.json` plus two alternating `selfcontact_<base>_edit0/1.json` payloads, next to the foot-lock
edits and shared with the app. The project needs a corrected result first (`correct`).

- `--frames` are inclusive result frames `A-B` (the numbers `review` reports), several separated by commas; files store
  them zero-based and end-exclusive. `edit --frames` replaces the intervals, `--add-frames`/`--remove-frames` change them.
- Per-track settings: `--distance` (palm-to-target metres, default auto), `--offset` (contact point in the target joint's
  frame, metres, default auto), `--reciprocal` (move the target limb too: `off`, `auto` when the arm cannot reach,
  `always`; default auto), `--grip` (`auto` turns the palm toward the target and wraps the fingers; default off).
- One hand holds one target at a time: two enabled tracks of the same hand may not overlap in frames.
- `suggest` runs the `review` missed-contact check (ExMR) on the uncorrected base result (`--result` to choose another)
  and stores each issue as suggested (`automatic`) intervals on the track for that hand and target, creating it if
  needed. Each run replaces all suggestions. `--accept` also adds them to the tracks' intervals, strongest issue first,
  skipping frames another enabled track already holds with the same hand; `edit --id ID --accept` accepts one track.
- `list` prints the tracks, intervals, suggestions, settings and, after `correct`, the last residuals per track.
- Every change gives the tracks a new revision, so the corrected result and its exports become stale until `correct`
  runs again (the command says so). `remove --all` deletes the tracks (foot, placement and smoothing rows are kept, see below); with nothing
  left the files are deleted and a result corrected without self-contacts is then
  current again. `--dry-run` validates a change without writing it.

### Foot articulation

```
rgbmocap-cli.cmd contacts feet list    --project P [--model M]
rgbmocap-cli.cmd contacts feet add     --project P --foot LeftFoot|RightFoot --mode pointe|flat|flexed|match --frames A-B[,C-D]
                                       [--strength S] [--profile default|dancer] [--id ID] [--disabled]
rgbmocap-cli.cmd contacts feet edit    --project P --id ID [--foot ...] [--mode ...] [--frames ...] [--add-frames ...]
                                       [--remove-frames ...] [--strength S] [--profile ...] [--enable|--disable]
rgbmocap-cli.cmd contacts feet remove  --project P --id ID | --all
```

An entry poses one foot's ankle and toes over frame intervals; `correct` applies it before the foot locks, so locks and
grounding see the articulated foot. Modes: `pointe` (foot in line with the shin, support on the toe box), `flat` (sole
level), `flexed` (dorsiflexed), `match` (fit the ankle to the video's 2D foot direction; ExMR only). `--strength` blends
from the source foot (0) to the full mode (1, default); `--profile dancer` allows plantarflexion up to 90 deg (default
50). One enabled mode per foot and frame. Entries live in the same `selfcontact_<base>_*.json` files as the tracks (ids
default to e.g. `lf-pointe`), use the same inclusive `--frames` and make the corrected result stale the same way. `feet
remove --all` keeps the tracks and `remove --all` keeps the foot entries, unless the files belong to motion with a
different length or frame rate: then `remove --all` deletes everything. `contacts list` also shows the entries and,
after `correct`, any skipped or joint-limited intervals. FlowHMR correction does not apply foot articulation.

### Limb placement and pose smoothing

```
rgbmocap-cli.cmd contacts place list    --project P [--model M]
rgbmocap-cli.cmd contacts place add     --project P --chain LeftArm|RightArm|LeftLeg|RightLeg --frames A-B[,C-D]
                                        [--strength S] [--id ID] [--disabled]
rgbmocap-cli.cmd contacts place edit    --project P --id ID [--chain ...] [--frames ...] [--add-frames ...]
                                        [--remove-frames ...] [--strength S] [--enable|--disable]
rgbmocap-cli.cmd contacts place remove  --project P --id ID | --all

rgbmocap-cli.cmd contacts smooth list   --project P [--model M]
rgbmocap-cli.cmd contacts smooth add    --project P [--group arms|legs|spine|all-written | --joints BONE[,BONE]] --frames A-B[,C-D]
                                        [--window N] [--id ID] [--disabled]
rgbmocap-cli.cmd contacts smooth edit   --project P --id ID [--group ... | --joints ...] [--window N] [--frames ...]
                                        [--add-frames ...] [--remove-frames ...] [--enable|--disable]
rgbmocap-cli.cmd contacts smooth remove --project P --id ID | --all
```

A limb placement row ("match 2D") fits one arm or leg chain to the video's 2D keypoints over frame intervals; `correct`
applies it after the hand lock and before the self-contacts, so contacts are solved on the placed limb. The upper joint
rotates freely and the elbow or knee bends about its hinge; the hand or foot keeps its world rotation and depth stays near
the source, since one view cannot measure it. `--strength` blends from the source limb (0) to the fitted one (1, default).
One enabled row per chain and frame; ids default to e.g. `rl-place`. Placement needs ExMR 2D keypoints.

A pose smoothing row calms rotation jitter of a bone group (`--group`, default `all-written`: the legs, plus the arms when
the hand lock moved them) or a bone list (`--joints`, any body bone except `Hips`) over frame intervals; `correct` applies
it last, after the self-contacts. `--window` is the Gaussian window in frames (3-61, default 5; keep it small). Leg joints
stay put while their foot is locked. Enabled rows sharing a bone may not overlap; ids default to e.g. `smooth-arms`.

Both use the same inclusive `--frames`, live in the same `selfcontact_<base>_*.json` files (`placement`, `smoothing`) and
make the corrected result stale the same way. `place remove --all` / `smooth remove --all` remove only their own rows and
`remove --all` keeps them (unless the files belong to other motion). `contacts list` shows them with, after `correct`,
each interval's 2D error before and after placement, skipped intervals and the largest smoothing change. FlowHMR
correction applies neither.

## Notes

- Projects, models and caches are the same as the app's; the CLI never deletes them.
- Multi-person Blender export of several takes is not available (the app's Export Checked People supports FBX/BVH only).
- MAMMA, PromptHMR and PEAR are not available from the command line.
