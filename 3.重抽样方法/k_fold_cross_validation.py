from random import seed
from random import randrange

def k_fold_cross_validation_split(dataset, folds=10):
    basket_for_splitted_data=list()
    fold_size=int(len(dataset)/folds)
    dataset_copy=list(dataset)
    
    for i in range(folds):
        basket_for_random_choosen_fold=list()
        while len(basket_for_random_choosen_fold)<fold_size:
            random_choose_some_element=randrange(len(dataset_copy))
            basket_for_random_choosen_fold.append(dataset_copy.pop(
                    random_choose_some_element))
        basket_for_splitted_data.append(basket_for_random_choosen_fold)
    return basket_for_splitted_data

seed(1)
dataset=[[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]]

k_folds_split=k_fold_cross_validation_split(dataset,5)

print(k_folds_split)