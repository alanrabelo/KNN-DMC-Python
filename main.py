
import random
import numpy as np
import collections



def euclideanDistance(instance1, instance2):

    distance = 0
    length = len(instance1)
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return distance

datasets = ['Datasets/iris.data', 'Datasets/column.arff']

for dataset in datasets:

    with open(dataset) as f:



        dataset = []
        categories = []

        for line in f.readlines():

            input = line.split(",")

            if input[-1] not in categories:
                categories.append(input[-1])

            input[-1] = list(categories).index(input[-1])
            dataset.append(input)

        for k in [1, 3, 5, 7, 9, 11, 13, 15]:

            number_of_corrects = 0
            number_of_errors = 0

            for execution in range(20):

                random.shuffle(dataset)

                split_index = round(len(dataset) * 0.8)
                train = dataset[:split_index]
                test = dataset[split_index:]

                input_size = len(train[0])

                train_x = np.array(train, dtype=float)[:, :-1]
                train_y = np.array(train, dtype=float)[:, -1]
                test_x = np.array(test, dtype=float)[:, :-1]
                test_y = np.array(test, dtype=float)[:, -1]

                for index,input in enumerate(test_x):

                    desired = test_y[index]

                    Ap = input  # "lowest point"
                    B = train_x  # sample array of points
                    dist = np.linalg.norm(B - Ap, ord=2, axis=1)  # calculate Euclidean distance (2-norm of difference vectors)
                    sorted_B = train_y[np.argsort(dist)]
                    closer_numbers = list(sorted_B)[:k]


                    counter = collections.Counter(closer_numbers)
                    closer = counter.most_common(1)[0][0]

                    if desired - closer == 0:
                        number_of_corrects += 1
                    else:
                        number_of_errors += 1

        print("Acertou %f" % ((number_of_corrects * 100) / (number_of_corrects + number_of_errors)))

