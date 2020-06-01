from math import sqrt
import data_reader


dataset=data_reader.read_csv('diabetes.csv')
for i in range(len(dataset[0])):
    data_reader.convert_string_to_float(dataset,i)
dataset=dataset[1:]

def find_the_max_and_min_in_the_dataset(dataset):
    max_and_min=list()
    for i in range(len(dataset[0])):
        col_values=[row[i] for row in dataset]
        value_max=max(col_values)
        value_min=min(col_values)
        max_and_min.append([value_max,value_min])
    return max_and_min

max_and_min=find_the_max_and_min_in_the_dataset(dataset)

#print(max_and_min)

def max_min_normalization(dataset,max_and_min):
    for row in dataset:
        for i in range(len(row)):
            row[i]=(row[i]-max_and_min[i][1])/(max_and_min[i][0]-max_and_min[i][1])
    

max_min_normalization(dataset,max_and_min)

#print(dataset)


def find_the_mean_of_the_dataset(dataset):
    means=list()
    for i in range(len(dataset[0])):
        col_values=[row[i] for row in dataset]
        mean=sum(col_values)/float(len(dataset))
        means.append(mean)
    return means

#a=find_the_mean_of_the_dataset(dataset)
#print(a)
    
def find_the_mean_of_the_dataset_V2(dataset):
    means=[888 for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        col_values=[row[i] for row in dataset]
        means[i]=sum(col_values)/float(len(dataset))
    return means
#a=find_the_mean_of_the_dataset(dataset)
#print(a)


def calculate_the_stdevs_of_the_dataset(dataset,means):
    stdevs=['William' for i in range(len(dataset[0]))]
    for i in range(len(dataset[0])):
        variance=[pow(row[i]-means[i],2) for row in dataset]
        stdevs[i]=sum(variance)
    stdevs=[sqrt(element/float(len(dataset)-1)) for element in stdevs]
    return stdevs

means_list=find_the_mean_of_the_dataset_V2(dataset)
stdevs_list=calculate_the_stdevs_of_the_dataset(dataset,means_list)
#print(dataset)
#print(means_list)
#print(stdevs_list)

def the_standardization_of_our_dataset(dataset,means_list,stdevs_list):
    for row in dataset:
        for i in range(len(row)):
            row[i]=(row[i]-means_list[i])/stdevs_list[i]

the_standardization_of_our_dataset(dataset,means_list,stdevs_list)

print(dataset)


           
#1.logarithm(exponential transfromation)
#2.power transfromation(Box-Cox)

    

    






