from KNN import *

datasets = ['Datasets/iris.data', 'Datasets/column.arff']

for dataset in datasets:
    knn = KNN(datasets[0])

    knn.load_data()
    knn.split_train_test_validation()
    knn.train(number_of_epochs=0)
