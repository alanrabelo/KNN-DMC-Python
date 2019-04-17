import random
import numpy as np
import collections
import matplotlib
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

            normalized_dataset = np.array(dataset, dtype=float)

            number_of_columns = len(dataset[0])
            for column in range(number_of_columns - 1):
                current_column = normalized_dataset[:, column]
                max_of_column = float(max(current_column))
                min_of_column = float(min(current_column))

                for index in range(len(dataset)):
                    value = dataset[index][column]
                    dataset[index][column] = (float(value) - min_of_column) / (max_of_column - min_of_column)

            self.dataset = dataset


    def split_train_test_validation(self):

        random.shuffle(self.dataset)

        validation_index = round(len(self.dataset) * (1 - VALIDATION_PERCENTAGE))

        self.k_folds_database = self.dataset[:validation_index]
        self.validation = self.dataset[validation_index:]


    def train(self, number_of_realizations):

        accuracies = []
        k_values = []

        for realization in range(1, number_of_realizations):

            k_accuracy = {}

            self.split_train_test_validation()
            NUMBER_OF_ELEMENTS_IN_FOLD = len(self.k_folds_database) / float(NUMBER_OF_FOLDS)

            for k in range(3,5, 2):

                for index in range(0,4):

                    dataset = np.array(self.k_folds_database, dtype=float)

                    start_index = round(index * NUMBER_OF_ELEMENTS_IN_FOLD)
                    final_index = round(start_index + NUMBER_OF_ELEMENTS_IN_FOLD)
                    TRAIN = np.array(list(dataset[:start_index, :]) + list(dataset[final_index:, :]))
                    TEST = dataset[start_index: final_index, :]

                    self.train_X = np.array(TRAIN, dtype=float)[:, :2]
                    self.train_Y = np.array(TRAIN, dtype=float)[:, -1]
                    self.test_X = np.array(TEST, dtype=float)[:, :2]
                    self.test_Y = np.array(TEST, dtype=float)[:, -1]

                    number_of_corrects = 0
                    number_of_errors = 0

                    confusion_matrix = {}

                    for index, input in enumerate(self.test_X):

                        desired = int(self.test_Y[index])

                        Ap = input  # "lowest point"
                        B = self.train_X  # sample array of points
                        dist = np.linalg.norm(B - Ap, ord=2,
                                              axis=1)  # calculate Euclidean distance (2-norm of difference vectors)
                        sorted_B = self.train_Y[np.argsort(dist)]
                        closer_numbers = list(sorted_B)[:k]

                        counter = collections.Counter(closer_numbers)
                        closer = int(counter.most_common(1)[0][0])

                        if desired - closer == 0:
                            number_of_corrects += 1
                        else:
                            number_of_errors += 1

                        if desired in confusion_matrix:
                            if closer in confusion_matrix[desired]:
                                confusion_matrix[desired][closer] += 1
                            else:
                                confusion_matrix[desired][closer] = 1
                        else:
                            confusion_matrix[desired] = {closer: 1}

                    accuracy_in_realization_with_k = number_of_corrects/(number_of_corrects + number_of_errors)

                    if k in k_accuracy:
                        k_accuracy[k].append(accuracy_in_realization_with_k)
                    else:
                        k_accuracy[k] = [accuracy_in_realization_with_k]

            for key in k_accuracy.keys():
                k_accuracy[key] = np.average(k_accuracy[key])

            self.best_k = sorted(k_accuracy, key=k_accuracy.get, reverse=False)[0]

            should_show_decision_surface = True if realization == 10 else False
            accuracies.append(self.test(should_show_decision_surface=should_show_decision_surface))
            k_values.append(self.best_k)

        print("The best K is %s" % max(set(k_values), key=k_values.count))

        self.plot_graphs(range(len(accuracies)), accuracies)

        return np.average(accuracies), np.std(accuracies)


    def test(self, should_show_decision_surface=False):

        validation_X = np.array(self.validation, dtype=float)[:, :2]
        validation_Y = np.array(self.validation, dtype=float)[:, -1]

        number_of_corrects = 0
        number_of_errors = 0

        for index, input in enumerate(validation_X):

            desired = validation_Y[index]

            Ap = input  # "lowest point"
            B = self.train_X  # sample array of points
            dist = np.linalg.norm(B - Ap, ord=2,
                                  axis=1)  # calculate Euclidean distance (2-norm of difference vectors)
            sorted_B = self.train_Y[np.argsort(dist)]
            closer_numbers = list(sorted_B)[:self.best_k]

            counter = collections.Counter(closer_numbers)
            closer = counter.most_common(1)[0][0]

            if desired - closer == 0:
                number_of_corrects += 1
            else:
                number_of_errors += 1

        return number_of_corrects / (number_of_corrects + number_of_errors)


    def predict(self, input):

        Ap = input  # "lowest point"
        B = self.train_X  # sample array of points
        dist = np.linalg.norm(B - Ap, ord=2,
                              axis=1)  # calculate Euclidean distance (2-norm of difference vectors)
        sorted_B = self.train_Y[np.argsort(dist)]
        closer_numbers = list(sorted_B)[:self.best_k]

        counter = collections.Counter(closer_numbers)
        closer = counter.most_common(1)[0][0]

        return closer

    def plot_decision_surface(self):

        # parameter_combination = list(itertools.combinations(range(len(list(X[0]))), 2))

        clear_red = "#FFA07A"
        clear_blue = "#B0E0E6"
        clear_green = "#98FB98"

        colors = [clear_red, clear_blue, clear_green]
        strong_colors = ['red', 'blue', '#2ECC71']

        number_of_points = 100
        for i in range(0, number_of_points, 1):
            for j in range(0, number_of_points, 1):

                x = i/number_of_points
                y = j/number_of_points
                value = int(self.predict([x, y]))

                color = colors[value]

                plt.plot([x], [y], 'ro', color=color)


        for index, input in enumerate(self.test_X):

            color_value = int(self.predict(input))
            plt.plot(input[0], input[1], 'ro', color=strong_colors[color_value])


        medium_colors = ['#641E16', '#1B4F72', '#186A3B']

        for index, input in enumerate(self.train_X):

            color_value = int(self.predict(input))
            plt.plot(input[0], input[1], 'ro', color=medium_colors[color_value])





        plt.show()

    def plot_graphs(self, X, Y):


        # plt.plot(X, Y)
        # plt.interactive(False)
        # plt.xticks(list(X))
        # plt.xlabel('Realizações')
        # plt.ylabel('Acurácia')
        plt.title(self.dataset_name)
        # plt.show()