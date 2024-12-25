import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout, Conv1D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

# Step 1: Load the dataset
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

def encode_data(train, test):
    # Encode species labels
    label_encoder = LabelEncoder().fit(train.species)
    labels = label_encoder.transform(train.species)
    classes = list(label_encoder.classes_)

    # Drop unnecessary columns
    train = train.drop(['species', 'id'], axis=1)
    test = test.drop('id', axis=1)

    return train, labels, test, classes

# Encode the data
train, labels, test, classes = encode_data(train, test)

# Step 2: Standardize the data
scaler = StandardScaler().fit(train.values)
scaled_train = scaler.transform(train.values)

# Split the dataset into training and validation sets
sss = StratifiedShuffleSplit(test_size=0.1, random_state=23)
for train_index, valid_index in sss.split(scaled_train, labels):
    X_train, X_valid = scaled_train[train_index], scaled_train[valid_index]
    y_train, y_valid = labels[train_index], labels[valid_index]

# Reshape data for Conv1D
nb_features = 64  # Number of features per feature type (shape, texture, margin)
nb_class = len(classes)

# Reshape training data
X_train_r = np.zeros((len(X_train), nb_features, 3))
X_train_r[:, :, 0] = X_train[:, :nb_features]
X_train_r[:, :, 1] = X_train[:, nb_features:128]
X_train_r[:, :, 2] = X_train[:, 128:]

# Reshape validation data
X_valid_r = np.zeros((len(X_valid), nb_features, 3))
X_valid_r[:, :, 0] = X_valid[:, :nb_features]
X_valid_r[:, :, 1] = X_valid[:, nb_features:128]
X_valid_r[:, :, 2] = X_valid[:, 128:]

# Convert labels to one-hot encoding
y_train = to_categorical(y_train, nb_class)
y_valid = to_categorical(y_valid, nb_class)

# Step 3: Define the CNN model
model = Sequential()
model.add(Conv1D(filters=512, kernel_size=1, input_shape=(nb_features, 3), activation='relu'))
model.add(Flatten())
model.add(Dropout(0.4))
model.add(Dense(2048, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(nb_class, activation='softmax'))

# Compile the model
sgd = SGD(learning_rate=0.01, nesterov=True, decay=1e-6, momentum=0.9)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Step 4: Train the model
nb_epoch = 15
history = model.fit(
    X_train_r, y_train,
    epochs=nb_epoch,
    validation_data=(X_valid_r, y_valid),
    batch_size=16,
    verbose=1
)

# Step 5: Evaluate the model
validation_loss, validation_accuracy = model.evaluate(X_valid_r, y_valid, verbose=1)
print(f"Validation Accuracy: {validation_accuracy * 100:.2f}%")
