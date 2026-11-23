import json, numpy as np
from pathlib import Path
import sys

ri = json.load(open('features/remote_index.json'))
tracks = ri['tracks']

human_ex = [(tid, t) for tid, t in tracks.items() if str(t.get('origin_label','')).lower()=='human'][0]
ai_ex = [(tid, t) for tid, t in tracks.items() if str(t.get('origin_label','')).lower() in ('ai','ai_generated','synthetic')][0]

for label, (tid, t) in [('human', human_ex), ('ai', ai_ex)]:
    bfp = t.get('beat_feature_path')
    tfp = t.get('times_feature_path')
    print(f'{label} {tid}:')
    print(f'  beat_feature_path: {bfp}')
    print(f'  times_feature_path: {tfp}')
    if bfp:
        p = Path(bfp)
        if not p.exists():
            p = Path('features') / p.name
        if p.exists():
            arr = np.load(str(p))
            print(f'  beat shape: {arr.shape}, dtype={arr.dtype}')
        else:
            print(f'  NOT FOUND')
    bc = t.get('beat_cache')
    if bc is not None:
        print(f'  beat_cache present: type={type(bc).__name__}, len={len(bc) if hasattr(bc,"__len__") else "N/A"}')
        if isinstance(bc, list) and len(bc) > 0:
            inner = bc[0]
            inner_len = len(inner) if isinstance(inner, list) else "scalar"
            print(f'  inner element: type={type(inner).__name__}, len={inner_len}')
    mc = t.get('mean_chroma')
    if mc is not None:
        print(f'  mean_chroma len={len(mc)}')

# Try to load a beat npy directly
print('\nSample beat .npy files:')
feat_dir = Path('features')
beat_files = sorted(feat_dir.glob('human_001*beat.npy'))
for f in beat_files[:3]:
    arr = np.load(str(f))
    print(f'  {f.name}: shape={arr.shape}')

beat_files2 = sorted(feat_dir.glob('ai_001*beat.npy'))
for f in beat_files2[:3]:
    arr = np.load(str(f))
    print(f'  {f.name}: shape={arr.shape}')
