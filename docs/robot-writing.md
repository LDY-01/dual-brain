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

The default paper is A4 portrait, 210 by 297 mm, with 8 mm margins. The call
response retains a 28 mm character cell. Canvas geometry uses the plan's paper
dimensions; A4 paper size does not establish physical robot reachability.

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
