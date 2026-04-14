# TensorFlow and tf.keras
import tensorflow as tf

fashion_mnist = tf.keras.datasets.fashion_mnist

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

x_train = x_train / 255.0
x_test = x_test / 255.0

# neuroniko diktyo
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(500, activation='sigmoid'), # 1o krimmeno epipedo me sigmoidi
    #tf.keras.layers.Dense(200, activation='sigmoid'), # 2o krimmeno epipedo me sigmoidi
    tf.keras.layers.Dense(10, activation='softmax') # 10 neurones eksodou
])

opt = tf.keras.optimizers.SGD(learning_rate=0.9) # SGD einai i Stochstic Gradient Descend

model.compile(optimizer=opt,loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10) # ekpaideysi toy neuronikou diktyou

test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2) #apeutheias vriskei to accuracy

print('\nTest accuracy:', test_acc)  # accuracy

