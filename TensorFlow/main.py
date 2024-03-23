import numpy as np
import random
import tensorflow as tf

# Example data from two files
with open('mots.txt', 'r') as file:
    file1_data = file.readlines()

with open('nombres.txt', 'r') as file:
    file2_data = file.readlines()


# Label the data: True for letters, False for numbers
labeled_data = [(string, True) for string in file1_data] + [(string, False) for string in file2_data]

# Shuffle the dataset
random.shuffle(labeled_data)

# Define function to convert characters to one-hot encoding
def char_to_one_hot(char):
    if char.isalpha():  # If character is a letter
        index = ord(char.lower()) - ord('a') + 1  # Map 'a' to 1, 'b' to 2, ..., 'z' to 26
        return [0] * index + [1] + [0] * (26 - index)
    else:  # If character is not a letter
        return [0] * 27  # 27th element for non-letter characters

# Convert strings to one-hot encoding
X = np.array([[char_to_one_hot(char) for char in string] for string, _ in labeled_data])
y = np.array([1 if label else 0 for _, label in labeled_data])

# Ensure all one-hot encodings have the same length
max_chars = max(len(string) for string, _ in labeled_data)
X_padded = np.array([string + [[0] * 27] * (max_chars - len(string)) if len(string) < max_chars else string for string in X])

# Flatten the input to have shape (None, 16 * 27)
X_flat = X_padded.reshape(X_padded.shape[0], -1)

# Split dataset into training and testing sets manually
split_index = int(0.8 * len(X_flat))  # 80% training, 20% testing
X_train, X_test = X_flat[:split_index], X_flat[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(max_chars * 27,)),  # Input shape: (max_chars * one_hot_encoding_size,)
    tf.keras.layers.Dense(8, activation='relu'),  # Hidden layer with 8 units
    tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer with 1 unit for binary classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy}')

# Make predictions (example)
predictions = model.predict(X_test[:5])
print("Example Predictions:", predictions.flatten())
