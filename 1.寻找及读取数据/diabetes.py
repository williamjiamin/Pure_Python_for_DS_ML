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

def convert_string_to_float(dataset_1,column):
    dataset_1=dataset_1[1:]
    for row in dataset_1:
        row[column]=float(row[column].strip())
#    print(dataset_1)
    

the_name_of_file_to_be_read='diabetes.csv'
dataset=read_csv(the_name_of_file_to_be_read)
#print(dataset)
dataset[0]

for i in range(len(dataset[0])):
    convert_string_to_float(dataset,i)

#print(dataset)
#print(dataset1)





        