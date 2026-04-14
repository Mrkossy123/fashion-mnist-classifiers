# Fashion MNIST Classification with k-NN, SVM and Neural Networks

Machine learning project in Python for image classification on the Fashion MNIST dataset.

## Περιγραφή
Η παρούσα εργασία αφορά την πειραματική μελέτη γνωστών μεθόδων ταξινόμησης πάνω στο σύνολο δεδομένων **Fashion MNIST**.

Το Fashion MNIST αποτελείται από εικόνες ρούχων διαστάσεων **28x28** grayscale και περιλαμβάνει:
- **60.000 εικόνες εκπαίδευσης**
- **10.000 εικόνες ελέγχου**
- **10 διαφορετικές κατηγορίες ενδυμάτων**

Στόχος της εργασίας είναι η εκπαίδευση και αξιολόγηση διαφορετικών ταξινομητών για την αναγνώριση της σωστής κατηγορίας κάθε εικόνας.

Στο repository περιλαμβάνονται υλοποιήσεις για:
- **k-Nearest Neighbors (k-NN)**
- **Support Vector Machines (SVM)**
- **Neural Networks**

## Υλοποιημένες Μέθοδοι

### 1. k-Nearest Neighbors (k-NN)
Στο αρχείο `kNN.py` υλοποιείται ταξινόμηση με τη μέθοδο k-NN με χρήση της βιβλιοθήκης `scikit-learn`.

Η διαδικασία περιλαμβάνει:
- φόρτωση του Fashion MNIST
- μετατροπή των εικόνων σε τύπο `float32`
- κανονικοποίηση τιμών pixel στο διάστημα `[0,1]`
- μείωση διαστάσεων με `block_reduce`, ώστε κάθε εικόνα να γίνει **7x7**
- μετατροπή κάθε εικόνας σε διάνυσμα 49 χαρακτηριστικών
- εκπαίδευση του ταξινομητή `KNeighborsClassifier`
- αξιολόγηση με **F1 score** και **Accuracy**

Στον παρόντα κώδικα χρησιμοποιείται:
- `n_neighbors = 10`
- ευκλείδεια απόσταση (`metric='euclidean'`)

### 2. Support Vector Machines (SVM)
Στο αρχείο `svm.py` υλοποιείται ταξινόμηση με χρήση **Support Vector Machine**.

Ακολουθείται παρόμοια προεπεξεργασία με αυτή του `kNN.py`:
- normalization
- block reduction
- flattening των εικόνων

Στη συνέχεια εκπαιδεύεται μοντέλο:
- `SVC(kernel="linear")`

και αξιολογείται στο test set με:
- **F1 score**
- **Accuracy**

### 3. Neural Network
Στο αρχείο `neural.py` υλοποιείται ένα νευρωνικό δίκτυο με χρήση **TensorFlow / Keras**.

Η αρχιτεκτονική που χρησιμοποιείται είναι:
- `Flatten` επίπεδο εισόδου για εικόνες 28x28
- ένα κρυφό επίπεδο με **500 νευρώνες** και ενεργοποίηση **sigmoid**
- επίπεδο εξόδου με **10 νευρώνες** και ενεργοποίηση **softmax**

Για την εκπαίδευση χρησιμοποιείται:
- βελτιστοποιητής **SGD**
- `epochs = 10`

Η απόδοση αξιολογείται στο test set με βάση το **accuracy**.

## Dataset
Το project χρησιμοποιεί το dataset **Fashion MNIST**.

Το dataset **δεν αποθηκεύεται τοπικά μέσα στο repository** ως `.csv` ή άλλο dataset file, επειδή φορτώνεται αυτόματα από:
- `keras.datasets.fashion_mnist`
- ή `tf.keras.datasets.fashion_mnist`

κατά την εκτέλεση των scripts.

Αυτό σημαίνει ότι κατά το πρώτο run απαιτείται σύνδεση στο internet για να γίνει λήψη των δεδομένων. Μετά το dataset αποθηκεύεται τοπικά και χρησιμοποιείται από εκεί.

## Αρχεία Repository
- `kNN.py` — υλοποίηση ταξινόμησης με k-NN
- `svm.py` — υλοποίηση ταξινόμησης με SVM
- `neural.py` — υλοποίηση νευρωνικού δικτύου
- `ML2021Homework1.pdf` — εκφώνηση εργασίας
- `README.md` — περιγραφή του project

## Απαιτήσεις
Για την εκτέλεση του project απαιτούνται:
- **Python 3**
- `numpy`
- `scikit-learn`
- `scikit-image`
- `tensorflow`

## Εγκατάσταση βιβλιοθηκών
Η εγκατάσταση μπορεί να γίνει με:

```bash
pip install numpy scikit-learn scikit-image tensorflow
