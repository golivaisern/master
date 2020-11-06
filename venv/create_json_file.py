import json
import os

f = open("/home/georgina/PycharmProjects/Master/venv/data_apartadoD.txt", "r")
result=list()

for x in f:
    res = json.loads(x)
    result.append(res)

print(result)
dir = '/home/georgina/PycharmProjects/Master/venv'
file_name = "data_apartadoC.json"

with open(os.path.join(dir, file_name), 'w') as file:
    file.write(json.dump(result, file))


