import csv
import os
print('Abrimos documentos extension csv')
csv_rows = []
csvfilename = ''
csv_file_object = open(csvfilename)
reader_obj = csv.reader(csv_file_object)
for row in reader_obj:
    if reader_obj.line_num == 1:
        continue #saltamos la primera linea
    csv_rows.append(row)
csv_file_object.close()

print('Leemos contenido documento csv')
csv_file_object = open(os.path.join('sin_cabeceras',csv_filename), 'w',newline='')
csv_writer = csv.writer(csv_file_object)
for rows in csv_rows:
    csv_writer.writerow(row)
csv_file_object.close()





