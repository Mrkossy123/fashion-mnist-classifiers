import numpy as np
import time
from sklearn import metrics
import skimage.measure
from keras.datasets import fashion_mnist
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()  # fortonei ta dedomena

# Normalize the data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0

################################################ Resize ############################################################
# FETOS OXI

resize_factor=4 # gia na ginei kathe eikona 7x7, diladi ena dianisma me 49 stoixeia

x_train_resized=[]
x_test_resized=[]

for i in range(0,len(x_train)):
    x_train_resized.append(skimage.measure.block_reduce(x_train[i], (resize_factor,resize_factor), np.average))

for i in range(0,len(x_test)):
    x_test_resized.append(skimage.measure.block_reduce(x_test[i], (resize_factor,resize_factor), np.average))

x_train = np.asarray(x_train_resized)
x_test = np.asarray(x_test_resized)

####################################################################################################################

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])
print(x_train.shape)
print(x_test.shape)

# x_train exei megethos 60000 (eikones) x 49 stoixeia i kathemia
# y_train exei megethos 60000 (eikones) x 1 stoixeio pou einai i katigoria tis eikonas
# x_test exei megethos 10000 (eikones) x 49 stoixeia i kathemia
# y_test exei megethos 10000 (eikones) x 1 stoixeio pou einai i katigoria tis eikonas

# IDEA: Prepei na ekpaideytei i kathe methodos sta (x_train,y_train), wste epeita
# na efarmostei i methodos sta (x_test,y_test) kai na sigrithoun oi katigories y_predict
# (pou prokiptoun gia ta x_test) me ta y_test gia na vrethoun ta accuracy, f1_score.

knn = KNeighborsClassifier(n_neighbors=10,metric='euclidean')
#knn = KNeighborsClassifier(n_neighbors=10,metric='cosine')

knn.fit(x_train, y_train) # ekpaideyetai o knn
y_pred_knn = knn.predict(x_test) # vgainoyn oi prvlepseis panw sta x_test


knn_f1 = metrics.f1_score(y_test, y_pred_knn, average= "weighted") # f1_score
knn_accuracy = metrics.accuracy_score(y_test, y_pred_knn) # accuracy

print("F1 score: {}".format(knn_f1))
print("Accuracy score: {}".format(knn_accuracy))
