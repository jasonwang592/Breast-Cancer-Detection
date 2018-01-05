import pandas as pd
import os
import subprocess
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

output = 'output/'
if not os.path.exists(output):
  os.makedirs(output)

df = pd.read_csv('files/Breast Cancer.csv')
diag_map = {'B': 'benign', 'M': 'malignant'}
df['diagnosis'] = df['diagnosis'].map(diag_map)
labs = df['diagnosis']
df.drop(['Unnamed: 32', 'id', 'diagnosis'], axis = 1, inplace = True)

#Split into train and test and fit decision trees
output_dir = output + 'decision_trees/'
if not os.path.exists(output_dir):
  os.makedirs(output_dir)
kf = KFold(n_splits = 5, shuffle = True, random_state = 42)
fold_accuracy = []
for train_indices, test_indices in kf.split(df):
  X_train, X_test = df.iloc[train_indices], df.iloc[test_indices]
  Y_train, Y_test = labs[train_indices], labs[test_indices]

  tree_model = tree.DecisionTreeClassifier(random_state = 42)
  tree_model.fit(X_train, Y_train)
  preds = tree_model.predict(X_test)
  accuracy = round((sum(preds == Y_test)/len(Y_test)) * 100, 3)
  print(' '.join(['Fold', str(len(fold_accuracy) + 1), 'Accuracy:', str(accuracy) + '%']))

  fname = ' '.join(['Decision Tree Fold', str(len(fold_accuracy) + 1)])
  with (open(output_dir + fname + '.dot', 'w')) as f:
    export_graphviz(tree_model, out_file = f,
                filled = True, rounded = True,
                special_characters = True,
                feature_names = df.columns)
  command = ['dot', '-Tpng', output_dir + fname + '.dot', '-o', output_dir + fname + '.png']
  subprocess.check_call(command)
  os.remove(output_dir + fname + '.dot')
  fold_accuracy.append(sum(preds == Y_test)/len(Y_test))
print(sum(fold_accuracy)/len(fold_accuracy))
