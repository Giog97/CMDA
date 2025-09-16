# find_broken_events.py
import h5py
import numpy as np
from dsec import DSECDataset

dataset = DSECDataset(dataset_txt_path='tuo/path.txt') # ./data/DSEC_Night/zurich_city_09_c/images/left/rectified/000658.png
broken_files = []

for i in range(len(dataset)):
    try:
        data = dataset[i]
        if 'events_vg' in data and data['events_vg'] is not None:
            if data['events_vg'].sum() == 0:  # Events tutti zero
                broken_files.append(dataset.dataset_txt[i][0])
    except Exception as e:
        broken_files.append(dataset.dataset_txt[i][0])
        print(f"Broken: {dataset.dataset_txt[i][0]} - {e}")

print(f"Found {len(broken_files)} broken event files")