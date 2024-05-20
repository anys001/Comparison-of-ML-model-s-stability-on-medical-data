import pickle
import sys
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.model_selection import train_test_split

TARGET_NAME_SRC = 'HeartDiseaseorAttack'
TARGET_NAME = 'Target'

SRC_FILE_PATH = sys.stdin.readline().rstrip()
datam = pd.read_csv(SRC_FILE_PATH, delimiter=',')

dis0 = datam[datam[TARGET_NAME_SRC].isin([0.0])]
dis1 = datam[datam[TARGET_NAME_SRC].isin([1.0])]
dis2 = dis0.sample(n=24000, random_state=42)
data = pd.concat([dis1, dis2])

data[TARGET_NAME] = data[TARGET_NAME_SRC].map({0.0: 0, 1.0: 1})

LABELS = { 0: "No Heart Diseaseor Attack", 1: "Heart Diseaseor Attack"}

FEATURES_CTG = {
  "Education": [1, 2, 3, 4, 5, 6]
  , "GenHlth": [1, 2, 3, 4, 5]
}

# For categorical features
for feature, categories in FEATURES_CTG.items():
    data[feature] = pd.Categorical(data[feature], categories=categories)
data = pd.get_dummies(data).astype('float32')

FEATURES = data.drop(columns=[TARGET_NAME_SRC, TARGET_NAME]).columns.tolist()

data_prepared = {
  "data": data,
  "features": FEATURES,
  "features_ctg_map": FEATURES_CTG,
  "labels": LABELS,
  "target": TARGET_NAME,
}

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

DST_FILE_PATH = sys.stdin.readline().rstrip()
save_object(data_prepared, DST_FILE_PATH)
print("<<<Done!>>>")
