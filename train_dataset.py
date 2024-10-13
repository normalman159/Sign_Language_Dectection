import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Load the data
data_train = pickle.load(open('./data.pickle', 'rb'))

# Check the structure of the data
data = data_train['data']
labels = data_train['labels']

# Ensure all elements in data have the same length
max_length = max(len(d) for d in data)
data = [np.pad(d, (0, max_length - len(d)), 'constant') if len(d) < max_length else d for d in data]

# Convert to NumPy arrays
data = np.asarray(data)
labels = np.asarray(labels)

# Split the data
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# Train the model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Predict and evaluate
y_predict = model.predict(x_test)
test_score = accuracy_score(y_test, y_predict)
print(f"Test score: {test_score*100}%")

# Save the model

f = open('model.pickle', 'wb')
pickle.dump({'model' : model}, f)
f.close()