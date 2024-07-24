import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

# Load the dataset with proper parameters
file_path = 'bank-additional-full.csv'  # Adjust the file path accordingly
# Assuming the file has a header row and is semicolon-separated with quoted headers
df = pd.read_csv(file_path, sep=';', quotechar='"')

# Check the structure of the dataset to ensure correct parsing
print(df.head())

# Verify column names to understand how they are loaded
print(df.columns)

# Encode categorical variables
cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan',
            'contact', 'month', 'day_of_week', 'poutcome']

label_encoders = {}
for col in cat_cols:
    label_encoders[col] = LabelEncoder()
    df[col] = label_encoders[col].fit_transform(df[col])

# Map 'y' column to binary values (assuming 'y' indicates purchase)
df['purchase_history'] = df['y'].map({'no': 0, 'yes': 1})

# Separate features and target variable
X = df.drop(columns=['y', 'purchase_history'])  # Features (excluding 'y' and 'purchase_history')
y = df['purchase_history']  # Target variable

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Predict on the test data
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Print confusion matrix and classification report
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
print('\nClassification Report:')
print(classification_report(y_test, y_pred))

# Optional: Visualize the Decision Tree
# Note: Visualizing large trees may be impractical; limit depth or use other methods

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, max_depth=3, feature_names=X.columns, class_names=['Not Purchased', 'Purchased'])
plt.title('Decision Tree Classifier')
plt.show()

