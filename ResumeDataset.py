import pandas as pd

# Download dataset from Kaggle
print("Loading dataset...")
full_path = 'D:/All_My_Learning/ML/UpdatedResumeDataSet.csv'


df = pd.read_csv(full_path)
print("Dataset loaded successfully!")
df.describe()
