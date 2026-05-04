from pathlib import Path
import os

BASE = '/content/SDTrack/SDTrack-Event'

# 1. Update train/admin/local.py — add illumination_dir_train
local_path = Path(f'{BASE}/lib/train/admin/local.py')
content = local_path.read_text()
if 'illumination_dir_train' not in content:
    content = content.replace(
        "self.eotb_dir_train = ",
        "self.illumination_dir_train = '/content/SDTrack/SDTrack-Event/data/train'\n        self.eotb_dir_train = "
    )
    local_path.write_text(content)
    Path(f'{BASE}/lib/train/admin/__pycache__/local.cpython-312.pyc').unlink(missing_ok=True)
    print("✓ train/admin/local.py updated")
else:
    print("✓ train/admin/local.py already updated")

# 2. Update base_functions.py — add Illumination to names2datasets
bf_path = Path(f'{BASE}/lib/train/base_functions.py')
content = bf_path.read_text()
if 'Illumination' not in content:
    content = content.replace(
        'from lib.train.dataset.FE108 import EOTB',
        'from lib.train.dataset.FE108 import EOTB\nfrom lib.train.dataset.illumination import Illumination'
    )
    content = content.replace(
        'assert name in ["EOTB_Train", "VISEVENT", \'FELT\']',
        'assert name in ["EOTB_Train", "VISEVENT", \'FELT\', \'Illumination\']'
    )
    content = content.replace(
        "        if name == \"EOTB_Train\":",
        "        if name == \"Illumination\":\n            datasets.append(Illumination(settings.env.illumination_dir_train, image_loader=image_loader))\n        if name == \"EOTB_Train\":"
    )
    bf_path.write_text(content)
    print("✓ base_functions.py updated")
else:
    print("✓ base_functions.py already updated")

print("\n✅ All done!")
