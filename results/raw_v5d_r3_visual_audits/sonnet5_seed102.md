# sonnet5_seed102 — visual audit sheet

## Spawn

- turn 1: [t001_0.jpg](../raw_v5d_r3/frames/sonnet5_seed102/t001_0.jpg)
- turn 1: [t001_1.jpg](../raw_v5d_r3/frames/sonnet5_seed102/t001_1.jpg)
- turn 1: [t001_2.jpg](../raw_v5d_r3/frames/sonnet5_seed102/t001_2.jpg)
- turn 1: [t001_3.jpg](../raw_v5d_r3/frames/sonnet5_seed102/t001_3.jpg)

## Doorways

_No logged frame at this event; review the linked filmstrip/video._

## Contacts

- turn 6: [t006_0.jpg](../raw_v5d_r3/frames/sonnet5_seed102/t006_0.jpg)
- turn 7: [t007_0.jpg](../raw_v5d_r3/frames/sonnet5_seed102/t007_0.jpg)
- turn 7: [t007_1.jpg](../raw_v5d_r3/frames/sonnet5_seed102/t007_1.jpg)
- turn 7: [t007_2.jpg](../raw_v5d_r3/frames/sonnet5_seed102/t007_2.jpg)
- turn 7: [t007_3.jpg](../raw_v5d_r3/frames/sonnet5_seed102/t007_3.jpg)
- turn 9: [t009_0.jpg](../raw_v5d_r3/frames/sonnet5_seed102/t009_0.jpg)
- turn 10: [t010_0.jpg](../raw_v5d_r3/frames/sonnet5_seed102/t010_0.jpg)
- turn 10: [t010_1.jpg](../raw_v5d_r3/frames/sonnet5_seed102/t010_1.jpg)
- turn 10: [t010_2.jpg](../raw_v5d_r3/frames/sonnet5_seed102/t010_2.jpg)
- turn 10: [t010_3.jpg](../raw_v5d_r3/frames/sonnet5_seed102/t010_3.jpg)
- turn 12: [t012_0.jpg](../raw_v5d_r3/frames/sonnet5_seed102/t012_0.jpg)
- turn 22: [t022_0.jpg](../raw_v5d_r3/frames/sonnet5_seed102/t022_0.jpg)
- turn 22: [t022_1.jpg](../raw_v5d_r3/frames/sonnet5_seed102/t022_1.jpg)
- turn 22: [t022_2.jpg](../raw_v5d_r3/frames/sonnet5_seed102/t022_2.jpg)
- turn 22: [t022_3.jpg](../raw_v5d_r3/frames/sonnet5_seed102/t022_3.jpg)
- turn 27: [t027_0.jpg](../raw_v5d_r3/frames/sonnet5_seed102/t027_0.jpg)
- turn 31: [t031_0.jpg](../raw_v5d_r3/frames/sonnet5_seed102/t031_0.jpg)
- turn 36: [t036_0.jpg](../raw_v5d_r3/frames/sonnet5_seed102/t036_0.jpg)

## Corrections

_No logged frame at this event; review the linked filmstrip/video._

## Kitchen Declare

_No logged frame at this event; review the linked filmstrip/video._

## Final

- turn 36: [t036_0.jpg](../raw_v5d_r3/frames/sonnet5_seed102/t036_0.jpg)

## Verdict

- locomotion: PASS — clear travel from bedroom spawn through doorway into the hallway (`results/videos_v5d_r3/sonnet5_seed102_filmstrip.png`, early/mid tiles; video ~t55)
- upright_trunk: FAIL — late video shows face-plant / heavily pitched trunk with a foot off the floor (`sonnet5_seed102.mp4` ~t60 and ~t72.4); not sustained upright standing
- alternating_feet_ground_clearance: PASS — alternating steps with swing-foot clearance during bedroom→hallway travel before collapse (filmstrip mid tiles; video ~t55)
- no_drag_glide_crawl_dither: FAIL — after collapse the body stays low with torso/head contact rather than a clean walk (video ~t60–t72; JSON late turns report head+torso `sustained_contact`)
- collision_no_teleport: PASS — motion remains continuous; wall/furniture contacts do not teleport the pose
- rooms_recognizable: PASS — bedroom (bed) and hallway geometry remain recognizable in egocentric stills + filmstrip
- metric_video_consistent: FAIL — video shows fall-like collapse (~t60/~t72.4) but JSON keeps `fell=False` through `timeout_turns` / turn_cap (path ~4.03 m, 11 bumps)
- reviewer: Cursor agent (filmstrip plus event-indexed stills + sampled mp4 frames)
- reviewed_at: 2026-08-03T16:35:00Z
