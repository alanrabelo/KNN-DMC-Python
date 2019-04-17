
import random
import numpy as np
import collections
import matplotlib.pyplot as plt


def euclideanDistance(instance1, instance2):

    distance = 0
    length = len(instance1)
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return distance


def intersection(lst1, lst2):
    lst3 = [list(filter(lambda x: x not in lst1, sublist)) for sublist in lst2]
    return lst3

datasets = ['Datasets/column.arff', 'Datasets/iris.data', 'Datasets/breast-cancer.data']
# datasets = ['Datasets/iris.data']

for dataset in datasets:

    with open(dataset) as f:

        dataset = []
        categories = []

        for line in f.readlines():

            input = line.replace("?", str(0)).split(",")

            if input[-1] not in categories:
                categories.append(input[-1])

            input[-1] = list(categories).index(input[-1])
            dataset.append(input)


        normalized_dataset = np.array(dataset)
        number_of_columns = len(dataset[0])
        for column in range(number_of_columns-1):
            current_column = normalized_dataset[:, column]
            max_of_column = float(max(current_column))
            min_of_column = float(min(current_column))

            for index in range(len(dataset)):
                value = dataset[index][column]
                dataset[index][column] = (float(value) - min_of_column) / (max_of_column - min_of_column)


        k_range = [1, 3, 5, 7, 9, 11, 13, 15]
        averages = []
        deviations = []

        for k in k_range:

            results = []

            for execution in range(20):

                number_of_corrects = 0
                number_of_errors = 0

                random.shuffle(dataset)
                number_of_folds = 5

                for fold in range(number_of_folds):

                    split_index = round(len(dataset) * (1 - (1.0/number_of_folds)))

                    initial_point = int(fold * 0.1)
                    final_point = initial_point + round(len(dataset) * (1.0/number_of_folds))

                    test = dataset[initial_point: final_point]

                    train = dataset[:initial_point] + dataset[final_point:]
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

                results.append((number_of_corrects * 100) / (number_of_corrects + number_of_errors))

            averages.append(np.average(results))
            deviations.append(np.std(results))
            # red dashes, blue squares and green triangles

        plt.title('Resultados do K-nn')
        fig_size = plt.rcParams["figure.figsize"]
        print(fig_size)
        fig_size[0] = 12
        fig_size[1] = 5
        plt.rcParams["figure.figsize"] = fig_size

        plt.subplot(1, 2, 1)
        plt.plot(k_range, np.array(averages), 'ro-')
        plt.xlabel('Valor de K')
        plt.ylabel('Acurácia (%)')

        plt.subplot(1, 2, 2)
        plt.plot(k_range, np.array(deviations) * 10, 'bo-')
        plt.xlabel('Valor de K')
        plt.ylabel('Desvio padrão (%)')

        plt.tight_layout()


        plt.show()






