# LRI Archive Inventory

Last updated: 2026-04-03

## Known LRI Collections

### /Volumes/Base Photos (mounted, accessible now)
- **1,081 LRI files** across 27 shooting days
- **Date range: 2018-01-19 through 2018-03-24**
- Also contains `.lris` files (Lumen state files — saved edits/depth work)
- Also contains `Mylio_a77711/` folder and `Verses_FullMovie.mp4`
- Organized by date folders (YYYY-MM-DD)

Breakdown by date:
```
2018-01-19:   10 LRI
2018-01-20:  165 LRI  ← largest single day
2018-01-24:   68 LRI
2018-01-25:   63 LRI
2018-01-28:   65 LRI
2018-01-29:   16 LRI
2018-01-30:   48 LRI
2018-02-01:   26 LRI
2018-02-04:   32 LRI
2018-02-05:   21 LRI
2018-02-08:   44 LRI
2018-02-09:   25 LRI
2018-02-10:    4 LRI
2018-02-12:   16 LRI
2018-02-13:   16 LRI
2018-02-18:   43 LRI
2018-02-20:  115 LRI  ← second largest day
2018-02-22:   46 LRI
2018-02-23:    8 LRI
2018-02-27:   47 LRI
2018-03-01:   20 LRI
2018-03-02:   66 LRI
2018-03-07:   13 LRI
2018-03-11:   25 LRI
2018-03-14:   18 LRI
2018-03-18:   26 LRI
2018-03-22:   31 LRI
2018-03-24:    4 LRI
```

### OWC STX HDD → SSD Recovery (in progress as of 2026-04-03)
- **7,890 files total, 1.04 TB**
- Recovery in progress: ~3h 15m remaining when last seen
- This is the main archive — will likely contain the bulk of the L16 catalog
- Destination path: TBD (update when recovery completes)

### ~/Documents/Light_Work/LRI/ (test samples)
- `L16_00177.lri`
- `L16_03632.lri`

---

## Notes on .lris Files

`.lris` files are Lumen state files — they store the user's depth edits, bokeh settings, crop, rating etc. for a corresponding `.lri`. They're separate from the raw capture data.

The `CIAPI::StateFileEditor` class in `libcp.dylib` handles these:
- `StateFileEditor::hasDepthEdits()` — check if user made depth edits
- `StateFileEditor::getThumbnail()` — extract preview
- `StateFileEditor::serialize/deserialize` — read/write

When batch processing, check for a matching `.lris` alongside each `.lri` and apply those settings if present.

---

## Estimated Total

- Base Photos: ~1,081 LRI
- OWC recovery: ~7,890 files (not all LRI — includes other formats)
- Rough estimate: **several thousand LRI files total**

Update this file once OWC recovery completes and destination is mounted.
