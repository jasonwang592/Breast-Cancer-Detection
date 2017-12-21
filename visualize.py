import seaborn as sns
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys

output = 'output/'
if not os.path.exists(output):
  os.makedirs(output)

df = pd.read_csv('files/Breast Cancer.csv')
diag_map = {'B': 'benign', 'M': 'malignant'}
df['diagnosis'] = df['diagnosis'].map(diag_map)
labs = df['diagnosis']
x = df.copy(deep = True)
x.drop(['Unnamed: 32', 'diagnosis', 'id'], axis = 1, inplace = True)

#split train and test data
train = df.sample(frac = 0.8, random_state = 123)
test = df.loc[~df.index.isin(train.index)]

#standardize the columns
x = (x - x.mean()) / x.std()

#Too many columns so we'll explore by splitting plots
chunks = [(1,10), (11,20), (21,30)]
output_dir = output + 'violinplots/'
if not os.path.exists(output_dir):
  os.makedirs(output_dir)
for i, chunk in enumerate(chunks):
  fname = '_'.join(['violinplot', str(i)])
  x_sub = x.iloc[:, chunk[0]:chunk[1]]
  x_sub = pd.concat([labs, x_sub], axis = 1)
  melted = pd.melt(x_sub,
    id_vars = 'diagnosis',
    var_name = 'feature',
    value_name = 'value')
  sns.set_style('whitegrid')
  fig = plt.figure(figsize = (15, 9))
  g = sns.violinplot(x = 'feature', y = 'value', hue = 'diagnosis', data = melted, palette = 'PRGn', split = True)
  g.set_xticklabels(g.get_xticklabels(), rotation = 45)
  plt.title('Feature distribution of malignant and benign breast cancer tumors')
  plt.tight_layout()
  plt.savefig(output_dir + fname)
  plt.close()

#Explore some bivariate KDE plots for like features
sns.set_style('darkgrid')
m = df.loc[df['diagnosis'] == 'malignant']
b = df.loc[df['diagnosis'] == 'benign']

f, ax = plt.subplots(figsize = (12, 8))

ax = sns.kdeplot(m['concavity_mean'], m['concave points_mean'], cmap = 'Reds', shade = True, shade_lowest = False)
ax = sns.kdeplot(b['concavity_mean'], b['concave points_mean'], cmap = 'Blues', shade = True, shade_lowest = False)

red = sns.color_palette('Reds')[-2]
blue = sns.color_palette('Blues')[-2]
ax.text(3.8, 4.5, 'malignant', size = 16, color = red)
ax.text(2.5, 8.2, 'benign', size = 16, color = blue)
plt.show()


