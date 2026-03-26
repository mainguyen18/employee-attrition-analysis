import json

with open('hr_employee_analysis.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    new_source = []
    for line in cell.get('source', []):
        line = line.replace('phát hiện được ~70% khách churn', 'phát hiện được nguy cơ nghỉ việc')
        line = line.replace('nguyên nhân churn tại Germany', 'mức độ nghỉ việc do overtime và Sales')
        line = line.replace('ảnh hưởng churn', 'ảnh hưởng nghỉ việc')
        line = line.replace('khách churn', 'nhân viên nghỉ việc')
        new_source.append(line)
    cell['source'] = new_source

with open('hr_employee_analysis.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook patched successfully.")
