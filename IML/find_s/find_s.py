import csv

def find_s_algorithm(file_path):

    with open(file_path, 'r') as f:
        data = list(csv.reader(f))


    hypothesis = [''] * (len(data[0]) - 1)

    for row in data:
        attributes = row[:-1]
        target = row[-1]

        if target.lower() == 'yes':

            if hypothesis[0] == '':
                hypothesis = attributes.copy()

            else:
                for i in range(len(attributes)):
                    if hypothesis[i] != attributes[i]:
                        hypothesis[i] = '?'

    return hypothesis


print("Final Hypothesis:", find_s_algorithm("data.csv"))
