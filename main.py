from KNN import *
from DMC import *

datasets = ['Datasets/iris.data', 'Datasets/column.arff', 'Datasets/artificial.data']

for dataset in datasets:
    knn = KNN(dataset)

    knn.load_data()
    knn.split_train_test_validation()
    averages, deviations = knn.train(number_of_realizations=25)

    print("KNN: para o dataset %s tivermos uma acurácia de %.2f e um desvio padrão de %.2f" % (dataset, averages, deviations))

    dmc = DMC(dataset)
    dmc.load_data()
    dmc.fit()
    averages, deviations = dmc.test()

    print("DMC: para o dataset %s tivermos uma acurácia de %.2f e um desvio padrão de %.2f" % (dataset, averages, deviations))


