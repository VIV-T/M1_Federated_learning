from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
test_size = 0.02
seed = 42

nutrition = fetch_ucirepo(id=887)
X = nutrition.data.features 
y = nutrition.data.targets 

# preprocess
int_col_list = ["RIAGENDR", "PAQ605", "DIQ010", "LBXGLU", "LBXGLT"]
X[int_col_list] = X[int_col_list].astype(int)
y['age_group'] = y['age_group'].map({'Adult': 1, 'Senior': 0}) # 'Adult' to 1 and 'Senior' to 0

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=seed, stratify=y
)

print(y_train.shape)
print(y_train)
