import numpy as np
import os

dir_path = '/home/henry/Documents/PythonProjects/nanoMOE/data/dolma3'
files = [
    'crime_and_law_val.bin',
    'finance_and_business_val.bin',
    'food_and_dining_val.bin',
    'games_val.bin',
    'health_val.bin',
    'literature_val.bin',
    'politics_val.bin',
    'religion_val.bin',
    'science_math_and_technology_val.bin'
]

TARGET_TOKENS = 4_000_768

for f in files:
    path = os.path.join(dir_path, f)
    arr = np.memmap(path, dtype=np.uint16, mode='r')
    trimmed = np.array(arr[:TARGET_TOKENS])

    # overwrite the file
    trimmed.astype(np.uint16).tofile(path)
    print(f'{f}: trimmed to {TARGET_TOKENS:,} tokens')
