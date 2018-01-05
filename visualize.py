import seaborn as sns
import pandas as pd
import os
import matplotlib.pyplot as plt
import plotters
from sklearn.decomposition import PCA
from tabulate import tabulate

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

#standardize the columns
x = (x - x.mean()) / x.std()
y = x.copy(deep = True)

#Plot violinplots for a high level view of each feature
output_dir = output + 'violinplots/'
plotters.violinplotter(x, labs,  output_dir)

#Choose features of interest and plot pairplots
output_dir = output + 'pairwiseplots/'
cols = ['diagnosis', 'symmetry_mean', 'fractal_dimension_mean']
# plotters.pairplots(df, cols, output_dir)

#Explore some bivariate KDE plots for like features
# plotters.bivariate_kde(df, 'concavity_mean', 'concave points_mean', output + 'kde_density_plots/')

output_dir = output + 'clustermaps/'
if not os.path.exists(output_dir):
  os.makedirs(output_dir)
#Looking at clustermaps to see what features are highly correlated and can be dropped
g = sns.clustermap(x.corr())
plt.setp(g.ax_heatmap.get_yticklabels(), rotation = 0)
plt.setp(g.ax_heatmap.get_xticklabels(), rotation = 90)
plt.savefig(output_dir + 'clustermap_original', bbox_inches='tight')
plt.close()

x.drop(['perimeter_mean', 'radius_mean', 'perimeter_worst','radius_worst',
        'area_worst', 'radius_se', 'perimeter_se'], axis = 1, inplace = True)
g = sns.clustermap(x.corr())
plt.setp(g.ax_heatmap.get_yticklabels(), rotation = 0)
plt.setp(g.ax_heatmap.get_xticklabels(), rotation = 90)
plt.savefig(output_dir + 'clustermap_drop1', bbox_inches='tight')
plt.close()

x.drop(['compactness_se', 'concavity_se', 'concave points_se','symmetry_se',
        'concavity_worst', 'compactness_worst', 'concave points_worst', 'texture_worst'], axis = 1, inplace = True)
g = sns.clustermap(x.corr())
plt.setp(g.ax_heatmap.get_yticklabels(), rotation = 0)
plt.setp(g.ax_heatmap.get_xticklabels(), rotation = 90)
plt.savefig(output_dir + 'clustermap_drop2', bbox_inches='tight')
plt.close()

#Preliminary PCA
plotters.cum_variance_explained(y, output + 'PCA/')

#PCA analysis post basic feature engineering
plotters.cum_variance_explained(x, output + 'PCA/')

pca = PCA(n_components = y.shape[1])
pca.fit(y)
x1 = pca.fit_transform(y)
x1 = pd.DataFrame(x1)
x1 = pd.concat([labs, x1], axis = 1)

for label, color in zip(('benign', 'malignant'), ('blue','red')):
  plt.scatter(x1[x1['diagnosis'] == label][0], x1[x1['diagnosis'] == label][1], label = label, c = color)
plt.savefig(output + 'PCA/' + 'Scatter, 2 components')

df = pd.DataFrame(pca.components_, columns=list(y.columns))

#Code to generate HTML table to put in blog post because Pandas is stupid
#and can't print tables in a nice way
header = []
ncomponents = 5
for i in range(ncomponents):
  header.append('PC' + str(i + 1))
print(tabulate(df.T.iloc[:, 0:ncomponents], headers = header, tablefmt = 'html'))

