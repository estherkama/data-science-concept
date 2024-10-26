
import pandas as pd
import numpy as np


# Load the dataset
data = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/road_accidents_sample.csv')
data.head()

# Save the model
with open('accident_severity_model.pkl', 'wb') as file:
    pickle.dump(model, file)



# Define dependent and independent variables
X = pd.get_dummies(data[['speed_limit', 'weather', 'road_type', 'traffic_density', 'time_of_day']])
y = data['accident_severity']

# Save the column names
model_columns = X.columns



# Encode categorical variables
sample_data_encoded = pd.get_dummies(sample_data)

# Align with training columns
sample_data_encoded = sample_data_encoded.reindex(columns=model_columns, fill_value=0)

# Predict using the saved model
predicted_severity = model.predict(sample_data_encoded)
print(f'Predicted Accident Severity: {predicted_severity[0]}')

from matplotlib import pyplot as plt
_df_2['speed_limit'].plot(kind='hist', bins=20, title='speed_limit')
plt.gca().spines[['top', 'right',]].set_visible(False)

from matplotlib import pyplot as plt
_df_12['weather'].plot(kind='line', figsize=(8, 4), title='weather')
plt.gca().spines[['top', 'right']].set_visible(False)

from matplotlib import pyplot as plt
_df_12['weather'].plot(kind='line', figsize=(8, 4), title='weather')
plt.gca().spines[['top', 'right']].set_visible(False)

import joblib
joblib.dump(model_columns, 'model_columns.pkl')

model_columns = joblib.load('model_columns.pkl')

# Prepare sample data
sample_data = pd.DataFrame({
    'speed_limit': [60],
    'weather': ['clear'],
    'road_type': ['highway'],
    'traffic_density': ['medium'],
    'time_of_day': ['day']
})
