# Light L16 Capture Improvements

## Philosophy

The camera's job is to capture the best possible raw LRI data.
All processing happens on the Mac. So improvements here focus entirely
on capture quality — not on-device rendering or preview quality.

---

## Highest Impact Improvements

### 1. Full Manual Exposure Control

**Why it matters:** The L16's auto-exposure makes independent decisions
per-module group. Even small exposure variance between the 28mm, 70mm,
and 150mm modules forces the fusion algorithm to reconcile differences,
introducing artifacts. Consistent manual exposure means:
- All 16 modules capture matching tonal values
- Fusion algorithm works on well-matched data
- Highlight protection (slight underexpose → recover in processing)
- No exposure hunting between shots in a sequence

**Where to look in openlight-camera APK:**
- Camera2 API calls for ISO and exposure time
- Settings/preferences that may be present but UI-hidden
- Look for `CaptureRequest.SENSOR_EXPOSURE_TIME` and
  `CaptureRequest.SENSOR_SENSITIVITY` calls

### 2. Manual Focus Lock

**Why it matters:** AF decisions affect depth map quality. The camera
picks a single focus distance for all modules. Manually locking focus
at the right depth before shooting:
- Ensures the primary subject is in the optimal stereo matching zone
- Eliminates AF hunting between shots
- Gives CIAPI cleaner stereo data to build depth maps from

### 3. Manual White Balance Lock

**Why it matters:** If AWB varies between modules or between shots in
a sequence, color fusion becomes messy. Locking WB to a fixed Kelvin
value (or a custom preset from a grey card) means:
- Consistent color across all 16 modules
- Cleaner input for the fusion pipeline
- Better color accuracy when processing on Mac

### 4. Disable On-Device Preview JPEG Generation

**Why it matters:** After each capture, the camera generates a preview
JPEG on-device. This wastes CPU cycles (heating the device, draining
battery) and adds post-capture lag before the next shot is ready.

Since all final processing happens on the Mac, the preview JPEG is
irrelevant. Disabling it or reducing it to minimum quality would:
- Reduce heat buildup during extended shoots
- Improve shot-to-shot speed
- Extend battery life

### 5. Exposure Bracketing Control

**Why it matters:** The camera supports bracketing internally.
Better UI access to bracket count and EV step would enable:
- 3-shot HDR captures (−2, 0, +2 EV)
- Our Mac pipeline (HDR-Transformer, DeepHDR) merges these
  far better than Light's 2019 approach
- Useful for high-contrast scenes where single exposure fails

### 6. Reduce Shutter Lag

**Why it matters:** Any delay between pressing the shutter and actual
capture means subject motion. The camera does pre-capture computation
that adds latency. Stripping unnecessary pre-processing from the
capture path gives more responsive capture.

---

## Lower Priority / Nice to Have

- **Timelapse mode** — periodic auto-capture for long sequences
- **GPS tagging toggle** — save battery if not needed
- **File naming control** — custom prefixes to distinguish shoots
- **Storage destination** — direct to external if USB-OTG works
- **Silent shutter** — disable the artificial shutter sound

---

## How to Approach the APK Modifications

### Starting Point
- [ookami125/openlight-camera](https://github.com/ookami125/openlight-camera)
  — already decompiled, patched for stability, repackaged
- Read this before writing any code — may already have some of these unlocked

### XDA Thread
- [XDA — Light L16 Firmware](https://xdaforums.com/t/light-l16-firmware.4403267/)
- Check all pages — community may have already done manual control patches
- Specifically search for "ISO", "manual", "exposure" in the thread

### Tools Needed
- `apktool` — decompile/recompile APK
- `jadx` — decompile to readable Java
- Android Studio — for rebuilding/signing
- ADB — sideload to camera (`adb install -r light_camera_modified.apk`)

### Signing
Modified APK needs to be signed before install. The camera may or may not
enforce signature verification beyond Android's standard check. If it does,
may need to root the device first.

### ADB Access
Connect via USB, enable developer options on the L16 (tap Build Number 7×),
enable USB debugging. Confirm with `adb devices`.

---

## What We Cannot Fix on the Camera Side

- **Rolling shutter within each module** — hardware limitation
- **Individual module sensor noise floor** — 2016-era sensors
- **Dynamic range per module** — compensate with bracketing instead
- **Processing speed for any on-device ML** — Snapdragon 820 is too slow;
  all neural work stays on Mac

---

## Testing a Capture Improvement

For each change to the camera app, test by:
1. Shooting the same static scene before and after the change
2. Transferring both LRIs to Mac
3. Processing through CIAPI
4. Comparing: exposure consistency, color consistency, sharpness, depth map quality

A grey card + color checker in frame makes exposure/WB comparisons objective.

---

## Related Docs

- `HANDOFF.md` — CIAPI processing pipeline (Mac side)
- `ANDROID_CAMERA.md` — hardware limits and ROM options
- `ARCHIVE_INVENTORY.md` — existing LRI collections
