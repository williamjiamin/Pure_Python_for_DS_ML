# Created by william from lexueoude.com. 更多正版技术视频讲解，
# 公众号1.乐学偶得（lexueoude）2.乐学FinTech (LoveShareFinTech)


from random import seed
from random import randrange
from random import random


def subsample(dataset, ratio=1.0):
    sample = list()
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = randrange(len(dataset))
        sample.append(dataset[index])
    return sample


def mean(numbers):
    result = sum(numbers) / float(len(numbers))
    return result


seed(1)
dataset = [[randrange(10)] for i in range(20)]

# print(dataset)

ratio = 0.10
for size in [1, 10, 100,1000,10000,100000,1000000]:
    sample_means = list()
    for i in range(size):
        sample = subsample(dataset, ratio)
        sample_mean = mean([row[0] for row in sample])
        sample_means.append(sample_mean)
    print("When sample is [%d],the estimated mean is [%.3f]" % (size, mean(sample_means)))

print("The real mean of our dataset is [%.3f]" % mean([row[0] for row in dataset]))
