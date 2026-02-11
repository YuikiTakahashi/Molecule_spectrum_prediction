import json

path = r'c:\Users\kyoto\Desktop\Github_repos\Molecule_spectrum_prediction\20260123_173_peak_fitting_Copy1.ipynb'
with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

nb['cells'][0]['source'] = ["print('hello')"]

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Added print("hello") to cell 1')
