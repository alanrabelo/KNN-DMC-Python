from KNN import *

datasets = ['Datasets/artificial.data', 'Datasets/iris.data', 'Datasets/column.arff']

for dataset in datasets:
    knn = KNN(dataset)

    knn.load_data()
    knn.split_train_test_validation()
    averages, deviations = knn.train(number_of_realizations=25)

    print("para o dataset %s tivermos uma acurácia de %.2f e um desvio padrão de %.2f" % (dataset, averages, deviations))
