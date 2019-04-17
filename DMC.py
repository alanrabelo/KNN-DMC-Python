
import random
import numpy as np
import collections
import matplotlib.pyplot as plt
import itertools

number_of_executions = 10

def euclideanDistance(instance1, instance2):

    distance = 0
    length = len(instance1)
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return distance


def intersection(lst1, lst2):
    lst3 = [list(filter(lambda x: x not in lst1, sublist)) for sublist in lst2]
    return lst3

def read_database(dataset_name):

    with open(dataset_name) as f:

        dataset = []
        categories = []

        for line in f.readlines():

            input = line.replace("?", str(0)).split(",")

            if input[-1] not in categories:
                categories.append(input[-1])

            input[-1] = list(categories).index(input[-1])
            dataset.append(input)




        if dataset_name == 'Datasets/breast-cancer.data':
            dataset = list(np.array(dataset)[:, 1:])

        normalized_dataset = np.array(dataset)

        number_of_columns = len(dataset[0])
        for column in range(number_of_columns - 1):
            current_column = normalized_dataset[:, column]
            max_of_column = float(max(current_column))
            min_of_column = float(min(current_column))

            for index in range(len(dataset)):
                value = dataset[index][column]
                dataset[index][column] = (float(value) - min_of_column) / (max_of_column - min_of_column)

        random.shuffle(dataset)
        number_of_folds = 5


        initial_point = int(0 * 0.1)
        final_point = initial_point + round(len(dataset) * (1.0 / number_of_folds))



        test = dataset[initial_point: final_point]

        train = dataset[:initial_point] + dataset[final_point:]

        train_x = np.array(train, dtype=float)[:, :-1]
        train_y = np.array(train, dtype=float)[:, -1]
        test_x = np.array(test, dtype=float)[:, :-1]
        test_y = np.array(test, dtype=float)[:, -1]

        return train_x, train_y, test_x, test_y, np.array(dataset, dtype=float)[:, :-1],  np.array(dataset, dtype=float)[:, -1]

class DMC:

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def fit(self, train_X, train_Y):

        self.train_x = train_X
        self.train_y = train_Y

        self.centroids_trainning = {}
        centroids = {}

        for index, x_value in enumerate(train_X):

            category = train_Y[index]

            if category in centroids:
                centroids[category].append(x_value)
            else:
                centroids[category] = [x_value]

        for key in centroids.keys():
            values = centroids[key]
            self.centroids_trainning[key] = sum(values) / len(values)

    def test(self, x_test, y_test):

        self.test_x = x_test
        self.test_y = y_test

        results = []

        number_of_corrects = 0
        number_of_errors = 0


        for index, input in enumerate(x_test):

            closest_distance = 99999999999999999
            closest_key = 0

            desired = y_test[index]

            for key in self.centroids_trainning.keys():

                distance = euclideanDistance(self.centroids_trainning[key], input)
                if distance < closest_distance:
                    closest_key = key
                    closest_distance = distance

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


    def plot_decision_surface(self):

        parameter_combination = list(itertools.combinations(range(len(list(X[0]))), 2))

        for parameter in parameter_combination:

            background_arrange = np.arange(0, 1, 0.1)
            print(list(itertools.combinations(background_arrange, 2)))



            fig_size = plt.rcParams["figure.figsize"]
            fig_size[0] = 7
            fig_size[1] = 7
            plt.rcParams["figure.figsize"] = fig_size

            for index, value in enumerate(self.X):

                colors = ['ro', 'bo', 'go']

                # for index, key in enumerate(dataset_dict):

                input = np.array(value)
                y = np.array(self.Y)[index]

                firstInput = input[parameter[0]]
                secondInput = input[parameter[1]]

                plt.plot(firstInput, secondInput, colors[int(y)])

            plt.xlabel('Superficie de Decisão')
            # plt.ylabel('Acurácia (%)')
            import uuid
            plt.savefig(str(uuid.uuid4()))
            plt.close()


datasets = ['Datasets/iris.data', 'Datasets/column.arff']
# datasets = ['Datasets/iris.data']

for dataset in datasets:

    all_averages = []
    all_deviations = []
    ploted = False

    for i in range(number_of_executions):

        train_x, train_y, test_x, test_y, X, Y = read_database(dataset)
        dmc = DMC(X, Y)

        dmc.fit(train_x, train_y)

        if not ploted:
            ploted = True
            dmc.plot_decision_surface()

        averages, deviations = dmc.test(test_x, test_y)
        all_averages.append(np.average(averages))
        all_deviations.append(np.average(deviations))

    plt.title('Resultados do DMC no %s' % dataset)
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 18
    fig_size[1] = 7
    plt.rcParams["figure.figsize"] = fig_size

    plt.subplot(1, 2, 1)
    plt.plot(range(len(all_averages)), np.array(all_averages), 'ro-')
    plt.xlabel('Execuções')
    plt.ylabel('Acurácia (%)')

    plt.subplot(1, 2, 2)
    plt.plot(range(len(all_deviations)), np.array(all_deviations) * 10, 'bo-')
    plt.xlabel('Execuções')
    plt.ylabel('Desvio padrão (%)')

    print("a acurácia média para o dataset %s é %.2f, e o desvio padrão foi %.2f" % (
        dataset, np.average(all_averages), np.average(all_deviations)))

    plt.show()
    plt.close()




