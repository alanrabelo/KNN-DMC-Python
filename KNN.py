import random
import numpy as np
import collections
import matplotlib.pyplot as plt

VALIDATION_PERCENTAGE = 0.1
TEST_PERCENTAGE = 0.2
NUMBER_OF_FOLDS = 5


class KNN:

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name

    def load_data(self):

        with open(self.dataset_name) as f:

            dataset = []
            categories = []

            for line in f.readlines():

                input = line.split(",")

                if input[-1] not in categories:
                    categories.append(input[-1])

                input[-1] = list(categories).index(input[-1])
                dataset.append(input)

            self.dataset = dataset

    def split_train_test_validation(self):

        random.shuffle(self.dataset)

        validation_index = round(len(self.dataset) * (1 - VALIDATION_PERCENTAGE))

        self.k_folds_database = self.dataset[:validation_index]
        self.validation = self.dataset[validation_index:]

    def train(self, number_of_epochs):

        accuracies = []
        k_values = []

        for realization in range(1, number_of_epochs):

            k_accuracy = {}

            self.split_train_test_validation()
            NUMBER_OF_ELEMENTS_IN_FOLD = len(self.k_folds_database) / float(NUMBER_OF_FOLDS)

            for k in range(1,15, 2):

                for index in range(0,4):

                    dataset = np.array(self.k_folds_database, dtype=float)

                    start_index = round(index * NUMBER_OF_ELEMENTS_IN_FOLD)
                    final_index = round(start_index + NUMBER_OF_ELEMENTS_IN_FOLD)
                    TRAIN = np.array(list(dataset[:start_index, :]) + list(dataset[final_index:, :]))
                    TEST = dataset[start_index: final_index, :]

                    train_X = np.array(TRAIN, dtype=float)[:, :-1]
                    train_Y = np.array(TRAIN, dtype=float)[:, -1]
                    test_X = np.array(TEST, dtype=float)[:, :-1]
                    test_Y = np.array(TEST, dtype=float)[:, -1]

                    number_of_corrects = 0
                    number_of_errors = 0

                    for index, input in enumerate(test_X):

                        desired = test_Y[index]

                        Ap = input  # "lowest point"
                        B = train_X  # sample array of points
                        dist = np.linalg.norm(B - Ap, ord=2,
                                              axis=1)  # calculate Euclidean distance (2-norm of difference vectors)
                        sorted_B = train_Y[np.argsort(dist)]
                        closer_numbers = list(sorted_B)[:k]

                        counter = collections.Counter(closer_numbers)
                        closer = counter.most_common(1)[0][0]

                        if desired - closer == 0:
                            number_of_corrects += 1
                        else:
                            number_of_errors += 1

                    accuracy_in_realization_with_k = number_of_corrects/(number_of_corrects + number_of_errors)

                    if k in k_accuracy:
                        k_accuracy[k].append(accuracy_in_realization_with_k)
                    else:
                        k_accuracy[k] = [accuracy_in_realization_with_k]

            for key in k_accuracy.keys():
                k_accuracy[key] = np.average(k_accuracy[key])

            self.best_k = sorted(k_accuracy, key=k_accuracy.get, reverse=False)[0]

            accuracies.append(self.test())
            k_values.append(self.best_k)

        print("The best K is %s" % max(set(k_values), key=k_values.count))
        self.plot_graphs(range(len(accuracies)), accuracies)

    def test(self):

        validation_X = np.array(self.validation, dtype=float)[:, :-1]
        validation_Y = np.array(self.validation, dtype=float)[:, -1]

        number_of_corrects = 0
        number_of_errors = 0

        for index, input in enumerate(validation_X):

            desired = validation_Y[index]

            Ap = input  # "lowest point"
            B = validation_X  # sample array of points
            dist = np.linalg.norm(B - Ap, ord=2,
                                  axis=1)  # calculate Euclidean distance (2-norm of difference vectors)
            sorted_B = validation_Y[np.argsort(dist)]
            closer_numbers = list(sorted_B)[:self.best_k]

            counter = collections.Counter(closer_numbers)
            closer = counter.most_common(1)[0][0]

            if desired - closer == 0:
                number_of_corrects += 1
            else:
                number_of_errors += 1

        return number_of_corrects / (number_of_corrects + number_of_errors)


    def plot_graphs(self, X, Y):

        plt.plot(X, Y)
        # plt.xticks(list(X))
        plt.xlabel('Realizações')
        plt.ylabel('Acurácia')
        plt.title(self.dataset_name)
        plt.show()