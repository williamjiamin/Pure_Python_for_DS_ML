from random import seed
from random import randrange



def train_test_split(dataset,train=0.60):
    train_lanzi=list()
    train_size=train*len(dataset)
    dataset_copy=list(dataset)
    
    while len(train_lanzi)< train_size:
        random_choose_some_element=randrange(len(dataset_copy))
        train_lanzi.append(dataset_copy.pop(random_choose_some_element))
    return train_lanzi, dataset_copy


seed(3)
dataset=[[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]

train , test = train_test_split(dataset)

print(train)
print(test)
        
        
    