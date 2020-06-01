def confusion_matrix(actual_data,prediected_data):
    unique_class_in_data=set(actual_data)
    matrix=[list() for x in range(len(unique_class_in_data))]
    for i in range(len(unique_class_in_data)):
        matrix[i]=[0 for x in range(len(unique_class_in_data))]
#    print(unique_class_in_data)
#    print(matrix)
    indexing_our_class=dict()
    for i ,class_value in enumerate(unique_class_in_data):
        indexing_our_class[class_value]=i
#    print(indexing_our_class)
    for i in range(len(actual_data)):
        col=indexing_our_class[actual_data[i]]
        row=indexing_our_class[prediected_data[i]]
        matrix[row][col] +=1
#        print(matrix)
    return unique_class_in_data , matrix
   
def pretty_confusion_matrix(unique_class_in_data,matrix):
    print('(Actual)   ' + ' '.join(str(x) for x in unique_class_in_data ))
    print('(Predicted)-------------------------')
    for i , x in enumerate(unique_class_in_data):
        print("{}  |       {}".format(x,  ' '.join(str(x) for x in matrix[i])))
        
    

actual_data=    [0,2,0,8,7,8,6,5,2,1]
prediected_data=[2,0,2,0,1,0,0,7,0,1]
unique_class_in_data,matrix =confusion_matrix(actual_data,prediected_data)
pretty_confusion_matrix(unique_class_in_data,matrix)

