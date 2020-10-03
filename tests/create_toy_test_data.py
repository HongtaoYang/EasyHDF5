from pathlib import Path
import numpy as np

a1 = np.ones([100, 100, 100], dtype=np.uint8)
a2 = np.ones([200, 200, 200], dtype=np.uint8) * 2
a3 = np.ones([300, 300, 300], dtype=np.uint8) * 3
a4 = np.ones([400, 400, 400], dtype=np.float32) * 1.1
a5 = np.ones([500, 500, 500], dtype=np.float32) * 2.2

for i, a in enumerate([a1, a2, a3, a4, a5]):
    save_dir = Path("tests/data/")
    save_dir.mkdir(parents=True, exist_ok=True)

    np.save(save_dir.joinpath(f"{i:02d}"), a)
