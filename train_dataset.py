import pickle
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split , RandomizedSearchCV
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
param_dist = {'n_estimators': randint(100,500),
              'max_depth': randint(1,20)}
model = RandomForestClassifier()
rand_search = RandomizedSearchCV(model, param_distributions=param_dist, n_iter=3, cv=5)
rand_search.fit(x_train, y_train)
best_model = rand_search.best_estimator_

# Predict and evaluate
y_predict = best_model.predict(x_test)
test_score = accuracy_score(y_test, y_predict)
print(f"Test score: {test_score*100}%")
print(f"Best model: {best_model}")
# Save the model

f = open('model.pickle', 'wb')
pickle.dump({'model' : best_model}, f)
f.close()