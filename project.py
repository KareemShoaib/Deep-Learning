import pandas as pd
import numpy as np
import os
import cv2
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Step 1: Load and Preprocess the Dataset
def load_and_preprocess_data():
    # Load CSV file
    train_df = pd.read_csv('train.csv')

    # Handle Missing Values
    numeric_columns = train_df.select_dtypes(include=['float64', 'int64']).columns
    train_df[numeric_columns] = train_df[numeric_columns].fillna(train_df[numeric_columns].mean())

    categorical_columns = train_df.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        train_df[col].fillna(train_df[col].mode()[0], inplace=True)

    # Remove Duplicates
    train_df.drop_duplicates(inplace=True)

    # Encode Labels
    label_encoder = LabelEncoder()
    train_df['species'] = label_encoder.fit_transform(train_df['species'])
    label_mapping = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))

    print(train_df)
    print("\nLabel Mapping:", label_mapping)

    # Load and Preprocess Images
    def load_images(image_folder, image_names, target_size=(128, 128)):
        images = []
        for name in image_names:
            img_path = os.path.join(image_folder, name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, target_size)
                images.append(img)
        return np.array(images)

    image_folder = 'images'  # Path to image folder
    train_images = load_images(image_folder, train_df['id'].astype(str) + '.jpg')
    train_images = train_images / 255.0  # Normalize to [0, 1]

    # Drop 'id' column as it's no longer needed
    train_df.drop('id', axis=1, inplace=True)

    # Split Dataset with Stratify
    X_train, X_test, y_train, y_test = train_test_split(
        train_images, train_df['species'].values, test_size=0.2, stratify=train_df['species'].values, random_state=42
    )

    num_classes = len(np.unique(y_train))  # Total number of classes

    return X_train, X_test, y_train, y_test, num_classes, label_mapping

# Step 2: Display Sample Images
def display_sample_images(X, y, label_mapping, num_samples=5):
    """
    Displays sample images along with their labels.

    Parameters:
    - X: Array of image data (shape: [num_images, height, width, channels])
    - y: Array of labels (integer encoded)
    - label_mapping: Dictionary mapping integer labels to species names
    - num_samples: Number of images to display
    """
    plt.figure(figsize=(15, 5))
    for i in range(num_samples):
        plt.subplot(1, num_samples, i + 1)
        plt.imshow(X[i])
        plt.title(f"Label: {label_mapping[y[i]]}")
        plt.axis('off')
    plt.show()

# Step 3: Training Function with L2 Regularization
def training(X_train, y_train, X_test, y_test, 
             batch_size=32, num_layers=3, dropout_rate=0.5, 
             optimizer_name='adam', weight_decay=0.01, 
             initial_lr=0.001, lr_scheduler=None, epochs=20):
    """
    Train a CNN model with specified hyperparameters and L2 regularization.
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3),
                     kernel_regularizer=l2(weight_decay)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    for _ in range(num_layers - 1):
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(weight_decay)))
        model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_regularizer=l2(weight_decay)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(y_train.max() + 1, activation='softmax', kernel_regularizer=l2(weight_decay)))

    if optimizer_name == 'adam':
        optimizer = Adam(learning_rate=initial_lr)
    elif optimizer_name == 'sgd':
        optimizer = SGD(learning_rate=initial_lr, momentum=0.9)
    elif optimizer_name == 'rmsprop':
        optimizer = RMSprop(learning_rate=initial_lr)
    else:
        raise ValueError("Invalid optimizer. Choose 'adam', 'sgd', or 'rmsprop'.")

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    callbacks = []
    if lr_scheduler:
        callbacks.append(LearningRateScheduler(lr_scheduler))

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )

    # Save the trained model using the recommended Keras format
    model.save('leaf_classification_cnn_model.keras')
    print("\nModel saved as 'leaf_classification_cnn_model.keras'")

    return history, model

# Step 4: Evaluation Function
def evaluation(model_path, X_train, y_train, X_test, y_test):
    """
    Load the trained model and evaluate its performance on the training and test sets.
    """
    model = tf.keras.models.load_model(model_path)
    print(f"\nModel loaded from {model_path}")

    # Recompile the model to rebuild metrics
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Evaluate on training set
    print("\nEvaluating on Training Set:")
    train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=0)
    print(f"Training Accuracy: {train_accuracy * 100:.2f}%")

    # Evaluate on test set
    print("\nEvaluating on Test Set:")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

    # Predictions on test set
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Classification Report
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred_classes))

# Step 5: Plot Training History
def plot_history(history, title=''):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy ' + title)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss ' + title)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Step 6: Run the Full Pipeline
if __name__ == "__main__":
    # Load and preprocess the data
    X_train, X_test, y_train, y_test, num_classes, label_mapping = load_and_preprocess_data()

    # Display some sample images from the training set
    print("\nDisplaying sample images from the training set...")
    display_sample_images(X_train, y_train, label_mapping, num_samples=5)

    # Train the model with L2 Regularization
    history, model = training(
        X_train, y_train, X_test, y_test,
        batch_size=32, num_layers=3, dropout_rate=0.5, 
        optimizer_name='adam', weight_decay=0.01, 
        initial_lr=0.001, lr_scheduler=None, epochs=20
    )

    # Plot training history
    plot_history(history, title='(Adam with L2 Regularization)')

    # Evaluate the model
    evaluation('leaf_classification_cnn_model.keras', X_train, y_train, X_test, y_test)
