
import random
import numpy as np
import collections
import matplotlib
import matplotlib.pyplot as plt
import itertools



number_of_executions = 1

def euclideanDistance(instance1, instance2):

    distance = 0
    length = len(instance1)
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return distance


def intersection(lst1, lst2):
    lst3 = [list(filter(lambda x: x not in lst1, sublist)) for sublist in lst2]
    return lst3


class DMC:

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

    def fit(self):

        random.shuffle(self.dataset)

        split_point = round(len(self.dataset)*0.8)

        train = np.array(self.dataset)[split_point:, :]
        test = np.array(self.dataset)[:split_point, :]

        self.train_x = train[:, :-1]
        self.train_y = train[:, -1]
        self.test_x = test[:, :-1]
        self.test_y = test[:, -1]

        self.centroids_trainning = {}
        centroids = {}

        for index, x_value in enumerate(self.train_x):

            category = self.train_y[index]

            if category in centroids:
                centroids[category].append(x_value)
            else:
                centroids[category] = [x_value]

        for key in centroids.keys():
            values = centroids[key]
            self.centroids_trainning[key] = sum(values) / len(values)

    def test(self):

        results = []

        confusion_matrix = {}

        number_of_corrects = 0
        number_of_errors = 0


        for index, input in enumerate(self.test_x):

            closest_distance = 99999999999999999
            closest_key = int(0)

            desired = int(self.test_y[index])

            for key in self.centroids_trainning.keys():

                distance = euclideanDistance(self.centroids_trainning[key], input)
                if distance < closest_distance:
                    closest_key = int(key)
                    closest_distance = distance


            if desired in confusion_matrix:
                if closest_key in confusion_matrix[desired]:
                    confusion_matrix[desired][closest_key] += 1
                else:
                    confusion_matrix[desired][closest_key] = 1
            else:
                confusion_matrix[desired] = {closest_key: 1}



            if desired - closest_key == 0:
                number_of_corrects += 1
            else:
                number_of_errors += 1

            results.append((number_of_corrects * 100) / (number_of_corrects + number_of_errors))

        return np.average(results), np.std(results)


    def predict(self, x):

        closest_distance = 99999999999999999
        closest_key = 0

        for key in self.centroids_trainning.keys():

            distance = euclideanDistance(self.centroids_trainning[key], x)
            if distance < closest_distance:
                closest_key = key
                closest_distance = distance

        return closest_key


    def plot_decision_surface(self, train_x, train_y, test_x, test_y):

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


        for index, input in enumerate(test_x):

            color_value = int(self.predict(input))
            plt.plot(input[0], input[1], 'ro', color=strong_colors[color_value])


        medium_colors = ['#641E16', '#1B4F72', '#186A3B']

        for index, input in enumerate(train_x):

            color_value = int(self.predict(input))
            plt.plot(input[0], input[1], 'ro', color=medium_colors[color_value])


        plt.show()




