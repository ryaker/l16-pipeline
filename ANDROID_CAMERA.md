# Light L16 Android Camera — Hardware Limits & Improvement Options

## The Hardware Reality

The L16 runs **Android 6.0.1 Marshmallow (AOSP)** on a **Snapdragon 820** (2016).
That SoC is the hard ceiling for on-device work. But there's a key insight:

> **The camera's job is capture. The M4's job is processing.**

Don't fight the 2017 hardware for computation — use the camera for what it does
uniquely well (simultaneous 16-module capture) and offload everything else.

---

## What Can Be Improved On-Device

### Camera App (light_camera.apk)
The APK has been extracted, decompiled, and repackaged by the community:
- [ookami125/openlight-camera](https://github.com/ookami125/openlight-camera)

Practical improvements possible without custom ROM:
- **Expose hidden settings** — shutter speed, ISO, white balance overrides
  that may be locked in the UI but exist in the code
- **Fix known bugs** — the community has patched crashes and stability issues
- **Disable forced processing** — stop the on-device preview computation to
  reduce heat/battery drain and just capture raw
- **Longer exposure bracketing** — the camera supports it internally

### Firmware
- Latest official: **1.3.51** — includes HD video, on-device editing, no-tripod
  low-light mode
- Archived at: https://archive.org/details/lfc-1351-0-00-ww-a-01-update
- Also: https://l16update.com/ (community update server, status unknown)

---

## What Cannot Be Fixed On-Device

- **Snapdragon 820 processing speed** — neural inference is too slow on this chip
  for real-time use; anything beyond basic ISP is impractical
- **2GB RAM** — limits how much the camera app can do concurrently
- **Battery life** — 16 modules + ISP + Android = short sessions; carry spares
- **Android 6.0.1** — too old for modern ML frameworks (TFLite needs Android 7+
  for hardware acceleration)

---

## What's Realistically Improvable via Custom ROM

The XDA community has investigated this:
- https://xdaforums.com/t/light-l16-firmware.4403267/

**Challenges:**
- Bootloader unlock status unclear
- Qualcomm BSP drivers are proprietary — custom kernel needs them
- AOSP 6.0.1 is ancient; jumping to a newer Android base requires porting all
  the camera HAL code (this is where the L16's custom hardware integration lives)

**Realistic outcome:** minor AOSP modifications possible (adb, root, disable OTA
check, expose developer options). Full LineageOS-style ROM is a major project.

---

## The Better Strategy: Camera as Dumb Capture Device

Instead of fighting the Android side, treat the L16 purely as a capture device:

1. **Disable/minimize on-device processing** — let it write raw LRI fast
2. **Use manual mode** — lock ISO/shutter to avoid per-shot variation
3. **Transfer LRIs to Mac immediately** via USB or WiFi
4. **All processing on M4** via our CIAPI wrapper + modern AI pipeline

This gets you:
- 2025-era processing on 2017-era capture hardware
- No dependency on the aging Android stack for quality
- Batch processing overnight for the backlog

---

## If You Find Both Cameras

Two cameras opens interesting options:
- **Stereo rig** — mount them side by side for wider baseline stereo
  (even more depth accuracy than single L16)
- **Backup capture** — one for active shooting, one preserved
- **Parts** — if one has dead modules, the other can still produce full output

---

## Community Resources

- [helloavo/Light-L16-Archive](https://github.com/helloavo/Light-L16-Archive)
- [ookami125/openlight-camera](https://github.com/ookami125/openlight-camera)
- [XDA — Light L16 Firmware thread](https://xdaforums.com/t/light-l16-firmware.4403267/)
- [archive.org — Latest OTA 1.3.51](https://archive.org/details/lfc-1351-0-00-ww-a-01-update)
