import seaborn as sns
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys
import plotters


output = 'output/'
if not os.path.exists(output):
  os.makedirs(output)

df = pd.read_csv('files/Breast Cancer.csv')
diag_map = {'B': 'benign', 'M': 'malignant'}
df['diagnosis'] = df['diagnosis'].map(diag_map)
df.drop(['Unnamed: 32', 'id'], axis = 1, inplace = True)
labs = df['diagnosis']
x = df.copy(deep = True)
x.drop(['diagnosis'], axis = 1, inplace = True)

#split train and test data
train = df.sample(frac = 0.8, random_state = 123)
test = df.loc[~df.index.isin(train.index)]

#standardize the columns
x = (x - x.mean()) / x.std()

#Plot violinplots for a high level view of each feature
output_dir = output + 'violinplots/'
plotters.violinplotter(x, output_dir)

#Choose features of interest and plot pairplots
output_dir = output + 'pairwiseplots/'
cols = ['diagnosis', 'concavity_mean', 'concave points_mean', 'concave points_worst']
plotters.pairplots(df, cols, output_dir)

# print(df.corr())
# sns.clustermap(df.corr())
# plt.tight_layout()
# plt.show()
# sys.exit()
#Explore some bivariate KDE plots for like features
# plotters.bivariate_kde(df, 'concavity_mean', 'concave points_mean', output + 'kde_density_plots/', save = True)



