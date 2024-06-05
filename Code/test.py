from cdr import read_cdr_folders

cdr_data = read_cdr_folders('Data')
print(cdr_data)
print(len(cdr_data), "Patients")