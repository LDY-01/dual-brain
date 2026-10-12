# Robot Writing: Software-First Development

For the agreed goals, stage completion criteria, and current status, see the
[Korean development roadmap](brushpen-response-roadmap.md).

The target is a spoken question followed by a one-sentence Korean answer written
with a brush pen. Development starts without a connected robot.

## Current Milestone

Implemented:

- Korean text to ordered monoline strokes, including modern Hangul syllables,
  compatibility jamo, spaces, and `.`, `,`, `!`, `?`.
- Unicode NFC normalization before glyph generation.
- Paper-relative millimeter coordinates, automatic character-cell wrapping,
  margin checks, and explicit rejection of unsupported input and overflow.
- Resampled paths and separate pen-up travel, pen-down, ink, and pen-up operations.
- PNG, SVG, normalized 12-second SVG animation, and JSON exports.
- Offline tests for all 11,172 modern Hangul syllables and planning invariants.

The offline preview does not implement speech recognition, model-generated answers,
physical motion, ink pressure, or camera verification. M1 now adds simulation-only
pen-tip IK, sampled approach/writing/departure/return preflight and MuJoCo pose replay.
Virtual-actuator dynamics and hold stopping are now checked for the baseline profile;
physical motor behavior, tool mass/compliance, pressure and writing remain unvalidated.
See the dated [simulation record](writing/simulation.md) and [dynamics results](writing/dynamics.md).
On 2026-10-12, a bounded condition matrix and durable simulation job manager were
added. See [the two-step results](writing/2026-10-12-progress.md); human review and
physical validation remain pending.

```powershell
& '.\.venv\Scripts\python.exe' -X utf8 kwon_lab/tools/writing_simulation.py --text '네' --render
& '.\.venv\Scripts\python.exe' -X utf8 kwon_lab/tools/writing_simulation.py --suite
```

The default mount is an uncalibrated 60mm local-x tip offset, not a measured grip.
Override it with `--config kwon_lab/writing/config/virtual_mount.json` or a modified
copy. Reports are diagnostics, never executable hardware command streams. Pose
replay is not a dynamics rollout. The CLI starts at a configured virtual airborne
ready pose, checks the approach and exit, and returns to those exact joints.
Reaching that ready pose from a physical robot's boot/current state is NOT validated.
See [dated transit checks](writing/transit.md) for results and limits.

CLI default scope is `full_cycle`; use `--writing-only` only to reproduce the
older writing-segment check. The Python `preflight` API retains writing-only
behavior unless `transit=TransitSettings()` is supplied. Motion remains denied
in both modes, and a writable path can fail because its exit is unreachable.

```powershell
# Actual mj_step integration of virtual actuators, not qpos reference replay
& '.\.venv\Scripts\python.exe' -X utf8 kwon_lab/tools/writing_simulation.py --dynamics --text '네' --render
# Request a simulated position-hold stop 5 seconds into the trajectory
& '.\.venv\Scripts\python.exe' -X utf8 kwon_lab/tools/writing_simulation.py --dynamics --stop-after-s 5
```

Dynamics uses a 2x longer reference schedule by default. It preserves the model's
gravity, actuator gains and force limits; the original 1x schedule is rejected for
tip overspeed. `dynamics_validated=true` means ONLY virtual model validation under
the recorded conditions. Physical dynamics and pressure flags stay false.
Dynamics stop success has `status=stopped`, `stop_test_passed=true`, and
`trajectory_complete=false`. It is not a physical emergency-stop certification.
Single-case runs now also export `summary.json`. Python exit codes: 0 passed,
2 rejected, 130 stopped; a shell wrapper may normalize nonzero exit codes.

## Text Call Screen

```powershell
& '.\.venv\Scripts\python.exe' -X utf8 kwon_lab/tools/writing_chat.py --port 8765
```

Open `http://127.0.0.1:8765`. Submitting the Korean call `야` produces the fixed
answer `네` and draws its strokes on the canvas. Other input is rejected; this is
not model-generated conversation or speech recognition. A busy tab blocks new
input. Stop, retry, completion, explicit errors, and job JSON download are supported.

The app is loopback-only, uses no external API, and never sends robot commands.
Its timestamps use KST; downloaded geometry retains motion-denied and
calibration-required flags. History is limited to the current tab and is not
persisted across reloads. Animation time and virtual paper dimensions are not
physical measurements. See the [dated progress record](writing/README.md) for
test results and screenshots. If the port is occupied, choose another `--port`.

The user-facing paper is A4 landscape, 297 by 210 mm, with 8 mm margins. The call
response retains a 28 mm character cell. Canvas geometry uses the plan's paper
dimensions; A4 paper size does not establish physical robot reachability.

## Simulation Jobs (2026-10-12)

```powershell
& '.\.venv\Scripts\python.exe' -X utf8 kwon_lab/tools/writing_chat.py `
  --port 8765 --records outputs/writing/jobs
```

Open `http://127.0.0.1:8765/jobs`. Enter fixed Korean text and a character cell
size (8-40 mm) to run the existing full-cycle MuJoCo backend. This writes the
entered text, not an AI answer. `/` remains the independent 2D call preview.
Both pages stay loopback-only and no hardware backend is available.

One simulation job may run per manager; another tab's submission returns 409.
An OS lock gives one process ownership of the same record directory. Separate
record directories are not a global hardware lock. Use `--config` to supply a
complete uncalibrated simulation profile; every new job reruns full preflight.

Per UUID, records contain `job.json`, `plan.json`, `config.json`, `report.json`,
`events.jsonl` and, when samples exist, `trace.png`. The PNG projects recorded
actual TCP segments, not planned glyphs or simulated ink. Explicit human screen
review adds `review.json`; it never changes physical validation flags.
Job wall duration and physics time are separate. Virtual stroke counts are
trajectory bookkeeping, not proof of visible or readable ink.

On 2026-10-12, persistent sheet IDs/occupied rows, subsequent non-overlapping
placement, space-exhaustion blocking and explicit new-sheet confirmation were
implemented. `pages.json` retains current and replaced sheets; each job adds
`placement.json` and a reservation status. Layout now uses the whole A4 landscape
paper inside its 8 mm margins (281x194 mm), independently of the unchanged
robot candidate area. Initially that area was 142x72 mm; the 2026-10-12 follow-up
default admits the full layout for testing and moves the virtual origin to
(0.160, -0.070, 0.005) m. Only the 28 mm call/greeting sequence is verified;
corners, center and later rows remain rejected by IK. This is not calibrated
physical reach, and every execution still runs the complete safety checks.
Preview performs layout only, not kinematic authorization; every execution reruns
the full preflight. Font sizes are never silently reduced. Humans prepare, secure
and replace paper; no automatic paper-changing backend is provided.
See [paper-space policy and results](writing/paper-space.md) and
[2026-10-12 landscape layout and failure categories](writing/a4-landscape.md).
See [virtual paper-placement comparison and remaining limits](writing/landscape-reach.md).
Paper overflow returns HTTP 409 with category `paper_space`. Backend reach,
IK and actuator-control-range failures use job `failure_category=robot_reach`;
other safety failures use `robot_validation`, and execution/storage errors use
`execution_record`. Proven pre-integration rejection releases the unused row
without treating the paper as full. A change from portrait requires explicit
new-sheet confirmation; old records and their original dimensions are retained.

`GET /api/page` returns the current sheet. `POST /api/page/preview` takes `text`,
`character_mm` and an optional expected `page_id`, without reserving or executing.
`POST /api/page/reset` requires the current ID and literal `confirmed=true`; it
rejects active jobs and stale confirmations. The UI sends an expected sheet ID
with every job. Omitting it is retained only for API compatibility.
Settled partial stops retain their row. Only proven pre-integration rejection
releases unwritten space. Unknown failures, orphan reservations and legacy jobs
without occupancy require human sheet confirmation, never automatic reuse.

Rollouts still initialize at the configured virtual ready pose. Physical recovery
from a partial stop is not implemented. Stored rows and confirmation are logical
state, not physical ink detection or paper registration. M0 human review and M3
physical commissioning remain required.

Normal completion, stop, failure, uncertainty and interrupted restart are distinct.
Only a verified full virtual trajectory is `completed`. Shutdown requests a
cooperative stop; a crash/restart marks active records `interrupted`, never
automatically resumes them. Storage failure blocks additional jobs. Corrupt
history fails closed. A stop during execution uses position hold, not automatic
pen lift or a physically certified emergency stop.

The default cooperative wall timeout is 120 seconds, distinct from simulated
time. The UI lists stored jobs after reload and permits allowlisted artifact
downloads; arbitrary filesystem access and foreign Host/Origin remain denied.

```powershell
& '.\.venv\Scripts\python.exe' -X utf8 kwon_lab/tools/writing_evaluation.py `
  --output outputs/writing/condition-review
& '.\.venv\Scripts\python.exe' -X utf8 -m unittest discover -s tests -p 'test_writing*.py' -v
node tests/writing_jobs_ui.cjs
node tests/writing_paper_ui.cjs
```

Browser checks require Node.js, Playwright and installed Edge; use `NODE_PATH`
for the environment's module location and `WRITING_TEST_URL` for another server.

## Run

From the repository root in PowerShell:

```powershell
& '.\.venv\Scripts\python.exe' -X utf8 kwon_lab/tools/writing_preview.py `
  --text '오늘은 어떤 일을 도와드릴까요?'
```

The tool prints the absolute locations of these files:

| File | Purpose |
| --- | --- |
| `preview.png` | Static review of the planned writing |
| `preview.svg` | Vector geometry |
| `animation.svg` | Stroke order; open in a browser and reload to replay |
| `plan.json` | Portable paper geometry and pen-state operations |

The animation has a fixed presentation duration. It does not represent the robot's
execution speed or account for pen-up travel time. The preview stroke width is a
display setting, not a prediction of brush width or force.

Optional layout parameters:

```powershell
& '.\.venv\Scripts\python.exe' -X utf8 kwon_lab/tools/writing_preview.py `
  --text '함께 시작해봐요.' `
  --width-mm 210 --height-mm 297 --character-mm 18
```

These dimensions describe a virtual paper area. They are not validated SO-101
reachability bounds. Shorten the text or explicitly change the preview layout when
it does not fit; the planner does not silently shrink or crop an answer.

Tests:

```powershell
& '.\.venv\Scripts\python.exe' -X utf8 -m unittest discover -s tests -v
```

## Geometry Contract

`writing.plan_text(text, PaperSettings(...))` has no dependency on robot ports,
cameras, MuJoCo, microphone streams, network access, or an LLM client. Exporting a
PNG additionally uses the existing Pillow dependency.

The JSON frame is the paper's upper-left origin, with x to the right, y down, and
z away from the paper. `z=0` is a virtual contact reference, not a motor target or
measured contact pressure. Each stroke ends with pen-up before the next travel.
The first travel assumes an executor can safely reach an approach position; it
does not define an approach from an arbitrary real robot pose.

Every exported plan records `motion_authorized=false` and
`physical_calibration_required=true`. No hardware executor is provided here.
An eventual real executor must independently enforce commissioning and runtime
safety checks; editing a JSON flag must never authorize motion.

The glyph library produces simple, legible centerline geometry. It is not a
learned handwriting or calligraphy model, and its stroke order is a development
font rather than a guarantee of traditional brush-writing technique. Digits,
Latin text, emoji, and unlisted punctuation currently fail explicitly.

## Next Milestones

The Korean roadmap defines the detailed checks and provisional acceptance
criteria. Its milestone IDs are the source of truth for progress tracking.

| ID | Milestone | Dependency |
| --- | --- | --- |
| M0 | Review Korean geometry, spacing, and stroke breaks | Preview implemented; human review pending |
| M1 | Pen-tip IK and safe robot trajectories in simulation | M0 |
| M2 | Fixed-text job control, stopping, and evaluation records | M0; validate with M1 |
| M3 | Physical mounting, calibration, cameras, and preflight | M1, M2, and available hardware |
| M4 | Safe air motion, contact height, and repeated ink lines | M3 |
| M5 | Repeated large Korean glyphs: ga, han, geul | M4 |
| M6 | Ten attempts of the same fixed sentence, blind human reading | M5; physical-writing gate |
| M7 | Typed questions to actual AI answers and physical writing | M6 |
| M8 | Windows microphone recording and Korean STT | M7 |
| M9 | End-to-end spoken-question evaluation | M8 |
| M10 | More sentences and paper-space management | M9 |

For now, prioritize M0-M2 without hardware. M3-M6 wait for hardware rather than
being marked complete from previews. Integrate answer generation and speech only
after M6 passes. If physical text is unreadable, fix mounting, calibration,
contact, or glyph geometry before adding AI functionality.

Physical glyph and sentence trials must record blind-reader transcriptions,
missing or merged strokes, execution failures, settings, and writing time. Keep
failed or interrupted attempts in the trial count. Passing one fixed sentence
does not establish readability for arbitrary generated answers.

When M7 starts, configure the answer provider and credentials separately from
the planner; never hard-code secrets or substitute canned replies for a working
model. M8 starts with explicit recording controls and a typed-input fallback;
add automatic speech detection after that workflow is reliable.

Changing from preview to physical execution requires a new backend and verified
calibration. It is not a flag change or a promise of simulation-to-real accuracy.
