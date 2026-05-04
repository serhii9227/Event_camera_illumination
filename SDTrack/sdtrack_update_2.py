from pathlib import Path

BASE = '/content/SDTrack/SDTrack-Event'

# 1. Rewrite test/evaluation/local.py
Path(f'{BASE}/lib/test/evaluation/local.py').write_text(f'''from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()
    settings.prj_dir = '{BASE}'
    settings.network_path = '{BASE}/pretrained_models'
    settings.result_plot_path = '{BASE}/output/test/result_plots'
    settings.results_path = '{BASE}/output/test/tracking_results'
    settings.save_dir = '{BASE}/output'
    settings.segmentation_path = '{BASE}/output/test/segmentation_results'
    settings.eotb_path = '{BASE}/data/test'
    return settings
''')
print("✓ local.py updated")

# 2. Update train/admin/local.py
train_local = Path(f'{BASE}/lib/train/admin/local.py')
train_local.write_text(train_local.read_text().replace('/data/users/xxx/SDTrack-Event', BASE))
Path(f'{BASE}/lib/train/admin/__pycache__/local.cpython-312.pyc').unlink(missing_ok=True)
print("✓ train/admin/local.py updated")

# 3. Update eotbdataset.py — universal for T=1, T=2, T=4
with open(f'{BASE}/lib/test/evaluation/eotbdataset.py', 'r') as f:
    content = f.read()
old = "    def _get_sequence_info_list(self):"
new = """    def _get_sequence_info_list(self):
        sequence_info_list = []
        stack_folder_priority = ['inter4_stack_3008', 'inter2_stack_3008', 'inter1_stack_3008']
        for seq_name in sorted(os.listdir(self.base_path)):
            seq_path = os.path.join(self.base_path, seq_name)
            if not os.path.isdir(seq_path):
                continue
            anno_path = os.path.join(seq_path, 'groundtruth_rect.txt')
            if not os.path.exists(anno_path):
                continue
            stack_name = None
            for candidate in stack_folder_priority:
                if os.path.exists(os.path.join(seq_path, candidate)):
                    stack_name = candidate
                    break
            if stack_name is None:
                continue
            stack_path = os.path.join(seq_path, stack_name)
            end_frame = len([f for f in os.listdir(stack_path) if f.endswith('_1.png')]) - 1
            sequence_info_list.append({
                'anno_path': f'{seq_name}/groundtruth_rect.txt',
                'endFrame': end_frame,
                'ext': 'png',
                'name': seq_name,
                'nz': 4,
                'object_class': 'object',
                'path': f'{seq_name}/{stack_name}',
                'startFrame': 0
            })
        return sequence_info_list

    def _get_sequence_info_list_old(self):"""
content = content.replace(old, new)
with open(f'{BASE}/lib/test/evaluation/eotbdataset.py', 'w') as f:
    f.write(content)
print("✓ eotbdataset.py updated")

# 4. Update loader.py
with open(f'{BASE}/lib/train/data/loader.py', 'r') as f:
    content = f.read()
content = content.replace('from torch._six import string_classes', 'string_classes = str')
with open(f'{BASE}/lib/train/data/loader.py', 'w') as f:
    f.write(content)
print("✓ loader.py updated")

# 5. Add weights_only=False
for filepath, old, new in [
    (f'{BASE}/lib/models/SDTrack/SDTrack.py',
     ', map_location="cpu")',
     ', map_location="cpu", weights_only=False)'),
    (f'{BASE}/lib/test/tracker/SDTrack.py',
     "torch.load(self.params.checkpoint, map_location='cpu')",
     "torch.load(self.params.checkpoint, map_location='cpu', weights_only=False)"),
    (f'{BASE}/lib/train/trainers/base_trainer.py',
     "torch.load(checkpoint_path, map_location='cpu')",
     "torch.load(checkpoint_path, map_location='cpu', weights_only=False)"),
]:
    with open(filepath, 'r') as f:
        content = f.read()
    with open(filepath, 'w') as f:
        f.write(content.replace(old, new))
print("✓ weights_only=False added")

print("\n✅ All done!")
