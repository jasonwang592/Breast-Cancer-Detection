import seaborn as sns
import pandas as pd
import os
import matplotlib.pyplot as plt
import sys
from sklearn.decomposition import PCA
import numpy as np


def cum_variance_explained(df, output_dir, save = True):
  '''Generates the cumulative variance explained plot for PCA based on number of components
  equal to the number of features
  Args:
    - df        (DataFrame): The dataframe all data (malignant and benign)
    - output_dir(String)   : The directory to save output to for bivariate_kde plots
    - save      (Boolean)  : Saves the file by default, if set to False, displays the plot instead
  '''

  pca = PCA(n_components = df.shape[1])
  pca.fit(df)
  var = pca.explained_variance_ratio_
  cumsum = np.cumsum(np.round(pca.explained_variance_ratio_, decimals = 4) * 100)
  plt.xlabel('Principal Components')
  plt.ylabel('Percentage of Variance Explained')
  plt.plot(cumsum)
  if save:
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    plt.savefig(output_dir + ','.join(['PCA Cumulative Variance', str(df.shape[1]) + ' components']))
  else:
    plt.show()
  plt.close()

def pairplots(df, features, output_dir, save = True):
  '''Plots pairplots for the data frame and desired features to visualize.

  Args:
    - df        (DataFrame): The dataframe all data (malignant and benign)
    - features  (list)     : List of feature names that will be pairplotted
    - output_dir(String)   : The directory to save output to for bivariate_kde plots
    - save      (Boolean)  : Saves the file by default, if set to False, displays the plot instead
  '''

  temp = df.loc[:, features]
  fname = ['pairplot']
  for i in features:
    if i != 'diagnosis':
      fname.append(i)
  sns.pairplot(data = temp, hue = 'diagnosis', palette = 'husl', diag_kind = 'kde')

  if save:
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    plt.savefig(output_dir + '+'.join(fname))
  else:
    plt.show()
  plt.close()

def bivariate_kde(df, feature1, feature2, output_dir, save = True):
  '''Plots the bivariate density plot of two features for malignant and
  benign breast cancer tumors.

  Args:
    - df        (DataFrame): The dataframe all data (malignant and benign)
    - feature1  (String)   : The first feature/metric to be compared
    - feature2  (String)   : The second feature/metric to be compared
    - output_dir(String)   : The directory to save output to for bivariate_kde plots
    - save      (Boolean)  : Saves the file by default, if set to False, displays the plot instead

  Raises:
    KeyError: If either feature1 or feature2 is not a valid column, raise an error.
  '''
  try:
    sns.set_style('darkgrid')
    m = df.loc[df['diagnosis'] == 'malignant']
    b = df.loc[df['diagnosis'] == 'benign']

    f, ax = plt.subplots(figsize = (12, 8))
    ax = sns.kdeplot(m[feature1], m[feature2], cmap = 'Reds', shade = True, shade_lowest = False)
    ax = sns.kdeplot(b[feature1], b[feature2], cmap = 'Blues', shade = True, shade_lowest = False)
  except KeyError as e:
    raise KeyError('Please provide a valid column feature to plot.')

  title = '\n'.join(['Disbribution of features in benign and malignant breast masses', feature1 + ' vs ' + feature2])

  red = sns.color_palette('Reds')[-2]
  blue = sns.color_palette('Blues')[-2]
  ax.text(m[feature1].mean() - 1.5 * m[feature1].std(), m[feature2].mean(), 'malignant', size = 16, color = red)
  ax.text(b[feature1].mean() - 2 * b[feature1].std(), b[feature2].mean(), 'benign', size = 16, color = blue)
  plt.title(title)
  plt.tight_layout()

  if save:
    if not os.path.exists(output_dir):
      os.makedirs(output_dir)
    plt.savefig(output_dir + '_'.join([feature1, feature2]))
  else:
    plt.show()
  plt.close()

def violinplotter(df, labs, output_dir, save = True):
  '''Helper function to make pairwise violinplots with split turned on to quickly compare high level distributions
  of features in breast masses examined.

  Args:
    - df        (DataFrame): The dataframe all data (malignant and benign)
    - labs      (DataFrame): DataFrame of labels
    - output_dir(String)   : The directory to save output to for violinplots
    - save      (Boolean)  : Saves the file by default, if set to False, displays the plot instead

  Raises:
    IndexError: Rasied if the DataFrame provided is not the full DataFrame of the dataset with all columns
  '''

  #Too many columns so we'll explore by splitting plots
  chunks = [(1,10), (11,20), (21,30)]
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)
  for i, chunk in enumerate(chunks):
    fname = '_'.join(['violinplot', str(i)])
    try:
      x_sub = df.iloc[:, chunk[0]:chunk[1]]
      x_sub = pd.concat([labs, x_sub], axis = 1)
      melted = pd.melt(x_sub,
        id_vars = 'diagnosis',
        var_name = 'feature',
        value_name = 'value')
      sns.set_style('whitegrid')
      fig = plt.figure(figsize = (15, 9))
      g = sns.violinplot(x = 'feature', y = 'value', hue = 'diagnosis', data = melted, palette = 'PRGn', split = True)
      g.set_xticklabels(g.get_xticklabels(), rotation = 45)
      plt.title('Feature distribution of malignant and benign breast masses')
      plt.tight_layout()
    except IndexError as e:
      raise IndexError('Please provide the full data frame with all 30 features/columns.')

    if save:
      if not os.path.exists(output_dir):
        os.makedirs(output_dir)
      plt.savefig(output_dir + fname)
    else:
      plt.show()
    plt.close()





