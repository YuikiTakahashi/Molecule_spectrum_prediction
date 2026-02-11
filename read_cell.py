import json

with open('C:/Users/kyoto/Downloads/Er_reproduce.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f'Total cells: {len(nb["cells"])}')
print(f'\nCell 24 type: {nb["cells"][24]["cell_type"]}')
print('\nCell 24 source:')
src = nb['cells'][24]['source']
content = ''.join(src) if isinstance(src, list) else src
print(content)
