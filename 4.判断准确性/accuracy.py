#Calculate accuracy of precdiction

def calculate_the_accuracy_of_prediction(actual_data, predicted_data):
	correct_num = 0
	for i in range(len(actual_data)):
		if actual_data[i] == predicted_data[i]:
			correct_num += 1
	return correct_num / float(len(actual_data))

actual_data=   [0,0,0,0,0,1,1,1,1,1]
predicted_data=[1,1,1,1,1,0,0,0,0,1]

accuracy=calculate_the_accuracy_of_prediction(actual_data,predicted_data)
print(accuracy)

