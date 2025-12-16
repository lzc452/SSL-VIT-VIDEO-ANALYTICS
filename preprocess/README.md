<!-- MP4
 → standardize_structure.py     （avi → mp4）
 → extract_frames.py            （video → frame folder）
 → generate_splits_lazy.py      （video → frame folder）
 → LazyFrameDataset             （on-the-fly sampling）
 → SSL / Finetune / Dynamic / Privacy / Federated
 -->

## Original Dataset Structure
```bash
data/
├─ UCF101/                # 101 classes
│   ├─ ApplyEyeMakeup/
|   |     ├─ v_ApplyEyeMakeup_g01_c01.avi
|   |     ├─ v_ApplyEyeMakeup_g01_c02.avi
|   |     └─ ...
│   ├─ ApplyLipstick/
|   |     ├─ v_ApplyLipstick_g01_c01.avi
|   |     ├─ v_ApplyLipstick_g01_c02.avi
|   |     └─ ...
│   ├─ Archery/
│   └─ ...
│
├─ HMDB51/                # 51 classes
│   ├─ brush_hair/
|   |     ├─ April_09_brush_hair_u_nm_np1_ba_goo_0.avi
|   |     ├─ April_09_brush_hair_u_nm_np1_ba_goo_1.avi
|   |     └─ ...
│   ├─ cartwheel/
|   |     ├─ (Rad)Schlag_die_Bank!_cartwheel_f_cm_np1_le_med_0.avi
|   |     ├─ Acrobacias_de_un_fenomeno_cartwheel_f_cm_np1_ba_bad_8.avi
|   |     └─ ...
│   └─ ...
│
├─ FaceForensics/          # 2 classes  (Real vs Fake)
│   ├─ fake/               # 200 videos
|   |    ├─ 01_02__outside_talking_still_laughing__YVGY8LOK.mp4
|   |    ├─ 01_02__walk_down_hall_angry__YVGY8LOK.mp4
|   |    └─ ...
│   └─ real/                # 200 videos
│       ├─ 01__exit_phone_room.mp4
│       ├─ 01__hugging_happy.mp4
│       ├─ 01__kitchen_pan.mp4
│       └─ ...
│
├─ Kinetics-400-Tiny/       # 400 classes
│   ├─ abseiling/               # 3 videos
|   |    ├─ _4YTwq0-73Y_000044_000054.mp4
|   |    ├─ _EDt9CNqqxk_000260_000270.mp4
|   |    └─ -3B32lodo2M_000059_000069.mp4
│   └─ air_drumming/            # 3 videos
│        ├─ _9SlRyQ2Aio_000192_000202.mp4
│        ├─ _axE99QAhe8_000026_000036.mp4
│        └─ _dbcJuKJQNs_000040_000050.mp4

```


## Standardized Dataset Structure
```bash
data/
├─ UCF101/                # 101 classes
│   ├─ ApplyEyeMakeup/
|   |     ├─ v_ApplyEyeMakeup_g01_c01.mp4
|   |     ├─ v_ApplyEyeMakeup_g01_c02.mp4
|   |     └─ ...
│   ├─ ApplyLipstick/
|   |     ├─ v_ApplyLipstick_g01_c01.mp4
|   |     ├─ v_ApplyLipstick_g01_c02.mp4
|   |     └─ ...
│   ├─ Archery/
│   └─ ...
│
├─ HMDB51/                # 51 classes
│   ├─ brush_hair/
|   |     ├─ April_09_brush_hair_u_nm_np1_ba_goo_0.mp4
|   |     ├─ April_09_brush_hair_u_nm_np1_ba_goo_1.mp4
|   |     └─ ...
│   ├─ cartwheel/
|   |     ├─ RadSchlag_die_Bank!_cartwheel_f_cm_np1_le_med_0.mp4
|   |     ├─ Acrobacias_de_un_fenomeno_cartwheel_f_cm_np1_ba_bad_8.mp4
|   |     └─ ...
│   └─ ...
│
├─ FaceForensics/          # 2 classes  (Real vs Fake)
│   ├─ fake/               # 200 videos
|   |    ├─ 01_02__outside_talking_still_laughing__YVGY8LOK.mp4
|   |    ├─ 01_02__walk_down_hall_angry__YVGY8LOK.mp4
|   |    └─ ...
│   └─ real/                # 200 videos
│       ├─ 01__exit_phone_room.mp4
│       ├─ 01__hugging_happy.mp4
│       ├─ 01__kitchen_pan.mp4
│       └─ ...
│
├─ Kinetics-400-Tiny/       # 400 classes
│   ├─ abseiling/               # 3 videos
|   |    ├─ _4YTwq0-73Y_000044_000054.mp4
|   |    ├─ _EDt9CNqqxk_000260_000270.mp4
|   |    └─ -3B32lodo2M_000059_000069.mp4
│   └─ air_drumming/            # 3 videos
│        ├─ _9SlRyQ2Aio_000192_000202.mp4
│        ├─ _axE99QAhe8_000026_000036.mp4
│        └─ _dbcJuKJQNs_000040_000050.mp4

```