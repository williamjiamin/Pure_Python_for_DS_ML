from csv import reader

def read_csv(the_name_of_file_to_be_read):
    dataset=list()
    with open(the_name_of_file_to_be_read,'r') as file:
        csv_reader=reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

def convert_string_to_float(dataset,column):
    dataset=dataset[1:]
    for row in dataset:
        row[column]=float(row[column].strip())
    

the_name_of_file_to_be_read='diabetes.csv'
dataset=read_csv(the_name_of_file_to_be_read)
#print(dataset)

for i in range(len(dataset[0])):
    convert_string_to_float(dataset,i)

print(dataset)





        