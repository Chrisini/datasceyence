#!/usr/bin/env python
# coding: utf-8

# # 𝔼𝕩𝕡𝕝𝕠𝕣𝕒𝕥𝕠𝕣𝕪 𝔻𝕒𝕥𝕒 𝔸𝕟𝕒𝕝𝕪𝕤𝕚𝕤 / 𝔻𝕠𝕞𝕒𝕚𝕟 𝕊𝕙𝕚𝕗𝕥
# 
# * Feature extraction
# * Feature selection (Correlation Matrix and (Random Forest or Univariate Feature Selection))
# * Plotting

# In[ ]:


# jupyter nbconvert --to script test_exploratory_analysis.ipynb


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap


def plot_examples(cms):
    """
    helper function to plot two colormaps
    """
    np.random.seed(19680801)
    data = np.random.randn(5, 5)

    fig, axs = plt.subplots(1, 2, figsize=(6, 3), constrained_layout=True)
    for [ax, cmap] in zip(axs, cms):
        psm = ax.pcolormesh(data, cmap=cmap, rasterized=True, vmin=-4, vmax=4)
        fig.colorbar(psm, ax=ax)
    plt.show()

N = 256
vals = np.ones((N, 4))
vals[:, 0] = np.linspace(0/256, 230/256, N)
vals[:, 1] = np.linspace(190/256, 140/256, N)
vals[:, 2] = np.linspace(190/256, 190/256, N)
variint_map3 = ListedColormap(vals)

N = 256
vals = np.ones((N, 4))
vals[:, 0] = np.linspace(0/256, 231/256, N)
vals[:, 1] = np.linspace(118/256, 238/256, N)
vals[:, 2] = np.linspace(118/256, 238/256, N)
variint_map1 = ListedColormap(vals)


N = 256
vals = np.ones((N, 4))
vals[:, 0] = np.linspace(243/256, 199/256, N)
vals[:, 1] = np.linspace(238/256, 21/256, N)
vals[:, 2] = np.linspace(237/256, 133/256, N)
variint_map2 = ListedColormap(vals)

top = cm.get_cmap(variint_map1, 128)
bottom = cm.get_cmap(variint_map2, 128)

newcolors = np.vstack((top(np.linspace(0, 0.8, 128)),
                       bottom(np.linspace(0.2, 1, 128))))

variint_map = ListedColormap(newcolors, name='variint_map')

plot_examples([variint_map3, variint_map])


# # Imports and Data

# In[10]:


import glob
import skimage.io
import numpy as np
import pandas as pd
import sklearn.model_selection
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
# from skimage.feature import greycomatrix, greycoprops
from skimage.util import img_as_ubyte

import scipy.stats
from skimage.transform import rescale, resize, downscale_local_mean

#from colour import *

#domain1 = glob.glob('C:/snec_data/Result_Data/cirrus/enface/*_cube_z.tif')
#domain2 = glob.glob('C:/snec_data/Result_Data/plex/enface/*_cube_z.tif')
#domain3 = glob.glob('C:/Users/Prinzessin/projects/image_data/iChallenge_AMD_OD_Fovea_lesions/images_AMD/*.jpg')
# domain2 = glob.glob('C:/Users/Prinzessin/projects/image_data/iChallenge_AMD_OD_Fovea_lesions/images_Non-AMD/*.jpg')

dev_mode = False

domain = []
domain_name = []

import os

import random

root = r"/local/scratch/CATARACTS-videos-processed"

if True:
    tmp1 = glob.glob('/local/scratch/CATARACTS-videos-processed/train*/*.jpg')
    random.shuffle(tmp1)
    domain.append(tmp1[0:1000])
    
    tmp2 = []
    for idn in  ['01', '07', '14', '16', '19']:
        tmp2.extend(glob.glob(os.path.join(root, f"test{idn}/*.jpg")))
    
    random.shuffle(tmp2)
    domain.append(tmp2[0:1000])
    
    domain_name.append("train")
    domain_name.append("val")
    # domain_name.append("test")
elif True:
    domain.append(glob.glob('C:/Users/Prinzessin/projects/image_data/drac_oct/DR/images/train/*')[0:10])
    domain.append(glob.glob('C:/Users/Prinzessin/projects/image_data/OCTA-500/OCTA_3M/Projection Maps/OCT(FULL)/*')[0:10])
    domain.append(glob.glob('C:/Users/Prinzessin/projects/image_data/iChallenge_AMD_OD_Fovea_lesions/images_AMD/*.jpg')[0:10])
    domain_name.append("DRAC")
    domain_name.append("OCTA-500")
    domain_name.append("Fundus")
elif False: 
    domain.append(glob.glob("E:/Christina/Result_Data/CIRRUS_*/enface/*_cube_z.tif") )
    domain.append(glob.glob("E:/Christina/Result_Data/PLEX_*/enface/*/*_cube_z.tif")  )      
    domain_name.append("CIRRUS" )
    domain_name.append("PLEX" )
    
else:
    # domain.append(glob.glob("E:/Christina/Result_Data/CIRRUS_*/enface/*_cube_z.tif") [0:10] )
    domain.append(glob.glob("E:/Christina/Result_Data/CIRRUS_Normal/enface/*_cube_z.tif"))
    domain.append(glob.glob("E:/Christina/Result_Data/CIRRUS_Glaucoma/enface/*_cube_z.tif"))
    domain.append(glob.glob("E:/Christina/Result_Data/PLEX_*/enface/*/*_cube_z.tif")      ) 
    domain.append(glob.glob("E:/Christina/Result_Data/ADAM_*/fundus/*jpg")  )
    domain_name.append("CIRRUS Normal" )
    domain_name.append("CIRRUS Glaucoma" )
    domain_name.append("PLEX Normal" )
    domain_name.append("Fundus" )
    
    #domain4 = glob.glob("E:/Christina/Result_Data/FDA_ADAM_2_CIRRUS/*jpg")
    #domain5 = glob.glob("E:/Christina/Result_Data/FDA_PLEX_2_CIRRUS/*jpg")
    # domain6 = glob.glob("E:/Christina/Result_Data/MUNIT_ADAM_2_CIRRUS/*jpg")
    domain.append(glob.glob("E:/Christina/Result_Data/MUNIT_PLEX_2_CIRRUS/*jpg") )
    domain_name.append("PLEX-2-CIRRUS (MUNIT)" )
    
    domain.append(glob.glob("E:/Christina/Result_Data/FDA_PLEX_2_CIRRUS/*jpg") )
    domain_name.append("PLEX-2-CIRRUS (FDA)" )
    
    domain.append(glob.glob("E:/Christina/Result_Data/FDA_FUNDUS_2_CIRRUS/*jpg") )
    domain_name.append("FUNDUS-2-CIRRUS (FDA)" )
    
domain


# # Feature Extraction

# In[ ]:


def get_features(img, label=None):
    
    glcm = graycomatrix(img, distances=[1], 
                           angles=[0], symmetric=True, 
                           normed=True)
    
    #plt.figure()
    #plt.imshow(glcm.squeeze())
    
    # print(scipy.stats.describe(glcm.flatten()))
    
    # print(graycoprops(glcm, "contrast")[0][0].type)
        
    feature = {"contrast" :  graycoprops(glcm, "contrast")[0][0],
               "dissimilarity" : graycoprops(glcm, "dissimilarity")[0][0],
               "homogeneity" : graycoprops(glcm, "homogeneity")[0][0],
               "ASM" : graycoprops(glcm, "ASM")[0][0],
               "energy" :  graycoprops(glcm, "energy")[0][0],
               "correlation" : graycoprops(glcm, "correlation")[0][0],
               # "kurtosis" : scipy.stats.kurtosis(img.flatten()),
               # "skew" : scipy.stats.skew(img.flatten()),
               #"coarse" : coarseness(img, 5),
               # "coarse 2" : get_coarseness_tamura(img),
               "mean img" :  np.mean(img),
               #"contrast 2" : contrast(img),
               #"direction" : directionality(img),
               "label" : label
                }    
    
    return feature


features = []

for d_i, d in enumerate(domain):
    
    
    for i, path in enumerate(d):
        
        
        if False:
            print(i, path)
        else:
            #try:
                #image = skimage.io.imread(path, plugin='tifffile')
            #except:
            image = skimage.io.imread(path, as_gray=True)
            image = img_as_ubyte(image)
            
            image = resize(image, (256, 256), anti_aliasing=True)
            image = img_as_ubyte(image)
                
            #print(image.shape)

            tmp = get_features(image, label=d_i)
            
        features.append(tmp)

    
"""
for path in domain1a:
    if dev_mode:
        image = skimage.io.imread(path, as_gray=True)
        image = img_as_ubyte(image)
    else:
        image = skimage.io.imread(path, plugin='pil')
    features1a.append(get_features(image, label=1))
    
    
    
for path in domain1b:
    if dev_mode:
        image = skimage.io.imread(path, as_gray=True)
        image = img_as_ubyte(image)
    else:
        image = skimage.io.imread(path, plugin='pil')
    features1b.append(get_features(image, label=1))  
print("done domain 1")      
    

for path in domain2:
    if dev_mode:
        image = skimage.io.imread(path, as_gray=True)
        image = img_as_ubyte(image)
    else:
        image = skimage.io.imread(path, plugin='pil')
    features2.append(get_features(image, label=2))
print("done domain 2")    

for path in domain3:
    image = skimage.io.imread(path, as_gray=True)
    image = img_as_ubyte(image)
    features3.append(get_features(image, label=3))
print("done domain 3")    
    

for path in domain7:
    image = skimage.io.imread(path, as_gray=True)
    image = img_as_ubyte(image)
    features7.append(get_features(image, label=3))
print("done domain 3") 
"""


# ## Features

# In[ ]:


df = pd.DataFrame(features)


# In[ ]:


df = pd.DataFrame(features) #  + features3)


features_train, features_test, gt_train, gt_test = sklearn.model_selection.train_test_split( df.loc[:, df.columns != "label"], df["label"], test_size=0.5, random_state=42)

print(df.shape)
print(df.info())
print(df.head(10))
print(features_train.head(10))
print(gt_train.head(10))


# # Feature selection

# ## Correlation Matrix
# * Remove correlating features with Paerson correlation

# In[ ]:


f = plt.figure(figsize=(10, 10))
plt.matshow(features_train.corr(), fignum=f.number, cmap=variint_map, vmin=-1, vmax=1)
plt.xticks(range(features_train.select_dtypes(['number']).shape[1]), features_train.select_dtypes(['number']).columns, fontsize=14, rotation=90)
plt.yticks(range(features_train.select_dtypes(['number']).shape[1]), features_train.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16)
#plt.subplots_adjust(left=2)
# fig.tight_layout()
plt.savefig("example_results/correlation.png", dpi=1200)


# In[ ]:


features_train2, features_test2 = features_train.drop(
    columns=["dissimilarity", "homogeneity", "ASM"]), features_test.drop(columns=["dissimilarity", "homogeneity", "ASM"])

features_train2


# ## Random Forest
# * Feature importance with RF and Gini / Permutation

# In[ ]:


from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=0)
rf.fit(features_train, list(gt_train))

result = rf.predict(features_test)
print(result)
print(list(gt_test))
acc = accuracy_score(list(gt_test), result)
print(acc)

r = rf.score(features_test, gt_test)
print(r)

features_test.columns


# In[ ]:


from sklearn.metrics import accuracy_score 
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(random_state=0)
rf.fit(features_train2, list(gt_train))

result = rf.predict(features_test2)
print(result)
print(list(gt_test))
acc = accuracy_score(list(gt_test), result)
print(acc)

r = rf.score(features_test2, gt_test)
print(r)

features_test.columns


# ### Gini (mean decrease in impurity)

# In[ ]:


importances = rf.feature_importances_
std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

forest_importances = pd.Series(importances, index=features_train2.columns)

fig, ax = plt.subplots()
forest_importances.plot.barh(xerr=std, ax=ax, color="darkorange")
plt.gca().invert_yaxis()

ax.set_title("Feature importances using MDI")
ax.set_ylabel("feature")
ax.set_xlabel("mean decrease in impurity")
fig.tight_layout()
plt.savefig("example_results/gini.png", dpi=1200)


# ### Permutation importance (Mean Decrease in Accuracy)
# * https://scikit-learn.org/stable/modules/permutation_importance.html

# In[ ]:


from sklearn.inspection import permutation_importance


rf = RandomForestClassifier(random_state=10)
rf.fit(features_train2, list(gt_train))


result = permutation_importance(
    rf, features_test2, list(gt_test), n_repeats=10, random_state=42, n_jobs=2
)

print(list(gt_test))

print(result)

forest_importances = pd.Series(result.importances_mean, index=features_test2.columns)

fig, ax = plt.subplots()
forest_importances.plot.barh(xerr=result.importances_std, ax=ax, color="darkorange")
plt.gca().invert_yaxis()

ax.set_title("Feature importances using permutation on full model")
ax.set_xlabel("mean accuracy decrease")
ax.set_ylabel("feature")
fig.tight_layout()
plt.savefig("example_results/permutation.png", dpi=1200)


# ## Univariate feature selection
# * Feature significance with P-Values and chi square scores

# In[ ]:


# import packages
import pandas as pd

# import data
my_df = df.drop(columns=["dissimilarity", "homogeneity", "ASM"]) #, df.copy()

from sklearn.feature_selection import SelectKBest, chi2

X = my_df.drop(["label"], axis = 1)
y = my_df["label"]

feature_selector = SelectKBest(chi2, k = "all")
fit = feature_selector.fit(X,y)

p_values = pd.DataFrame(fit.pvalues_)
scores = pd.DataFrame(fit.scores_)

input_variable_names = pd.DataFrame(X.columns)
summary_stats = pd.concat([input_variable_names, p_values, scores], axis = 1)
summary_stats.columns = ["input_variable", "p_value", "chi2_score"]
summary_stats.sort_values(by = "p_value", inplace = True)

p_value_threshold = 0.05
score_threshold = 5

selected_variables = summary_stats.loc[(summary_stats["chi2_score"] >= score_threshold) &
                                       (summary_stats["p_value"] <= p_value_threshold)]
selected_variables = selected_variables["input_variable"].tolist()
X_new = X[selected_variables]

#print(X_new)


# forest_importances = pd.DataFrame(summary_stats["p_value"].flatten(), columns=features_test.columns)

forest_importances = summary_stats["p_value"].reindex(index=features_test2.columns)

index = features_test2.columns

forest_importances = pd.DataFrame({'p_value': list(summary_stats["p_value"].sort_index())}, index=index)

print(summary_stats["chi2_score"].sort_index())
print(summary_stats["p_value"].sort_index())


# ### p-value

# In[ ]:


fig, ax = plt.subplots()
ax.barh(features_test2.columns, 1 - summary_stats["p_value"].sort_index(), color="darkorange") # yerr=result.importances_std
plt.gca().invert_yaxis()
ax.set_title("Feature significance")
ax.set_xlabel("1 - p_value")
ax.set_ylabel("feature")
fig.tight_layout()
plt.savefig("example_results/one_minus_pvalue.png", dpi=1200)


# ### chi2 score

# In[ ]:


fig, ax = plt.subplots()
ax.barh(features_test2.columns, summary_stats["chi2_score"].sort_index(), color="darkorange") # yerr=result.importances_std
plt.gca().invert_yaxis()
ax.set_title("Feature significance")
ax.set_xlabel("chi2 score")
ax.set_ylabel("feature")
fig.tight_layout()
plt.savefig("example_results/chi2score.png", dpi=1200)


# ## Plotting 2 features

# In[ ]:



key_x = "dissimilarity"
key_y = "mean img"
# key_y = "energy"

key_x = "contrast"
#key_y = "energy"
#key_y = "correlation"
#key_y = "dissimilarity"
#key_y = "kurtosis"


plt.figure()
plt.xlabel(key_x)
plt.ylabel(key_y)


print(df.head())

try:
    i = 0
    x = df[df['label'] == i][key_x]
    y = df[df['label'] == i][key_y]
    plt.scatter(x, y, label=domain_name[i], color='teal', s=10, marker='^')
except:
    pass

try:
    i += 1
    x = df[df['label'] == i][key_x]
    y = df[df['label'] == i][key_y]
    plt.scatter(x, y, label=domain_name[i], color='black', s=10, marker='^')
except:
    pass
try:
    i += 1
    x = df[df['label'] == i][key_x]
    y = df[df['label'] == i][key_y]
    plt.scatter(x, y,  label=domain_name[i], color='blue', s=10, marker='^')
except:
    pass
try:
    i += 1
    x = df[df['label'] == i][key_x]
    y = df[df['label'] == i][key_y]
    plt.scatter(x, y,  label=domain_name[i], color='green', s=10, marker='^')
except:
    pass
try:
    i += 1
    x = df[df['label'] == i][key_x]
    y = df[df['label'] == i][key_y]
    plt.scatter(x, y, label=domain_name[i], color='mediumvioletred', s=10, marker='^')
except:
    pass
try:
    i += 1
    x = df[df['label'] == i][key_x]
    y = df[df['label'] == i][key_y]
    plt.scatter(x, y, label=domain_name[i], color='red', s=10, marker='^')
except:
    pass
try:
    i += 1
    x = df[df['label'] == i][key_x]
    y = df[df['label'] == i][key_y]
    plt.scatter(x, y, label=domain_name[i], color='darkorange', s=10, marker='^')
except:
    pass

plt.legend()
plt.tight_layout()
plt.savefig("example_results/domainshift.png", dpi=1200)


# In[ ]:


key_x = "dissimilarity"
key_y = "mean img"
# key_y = "energy"

# key_x = "kurtosis"
key_y = "correlation"
#key_y = "kurtosis"

try:
    plt.figure()
    plt.xlabel(key_x)
    plt.ylabel(key_y)
    x3 = df[df['label'] == 0][key_x]
    y3 = df[df['label'] == 0][key_y]
    plt.scatter(x3, y3, label=domain3_name, color='teal', s=7, marker=',')
    x1 = df[df['label'] == 1][key_x]
    y1 = df[df['label'] == 1][key_y]
    plt.scatter(x1, y1, label=domain1_name, color='mediumvioletred', s=10, marker='+')
    plt.scatter(x2, y2, label=domain2_name, color='darkorange', s=10, marker='^')
    x2 = df[df['label'] == 2][key_x]
    y2 = df[df['label'] == 2][key_y]
    

except:
    pass

plt.legend()
plt.tight_layout()
plt.savefig("example_results/domainshift2.png", dpi=1200)


# # Mean and average images
# * https://towardsdatascience.com/exploratory-data-analysis-ideas-for-image-classification-d3fc6bbfb2d2

# In[ ]:


def img2np(paths, size = (150, 150)):
    # making n X m matrix
    # iterating through each file

    images = []
    current_image = None
    
    for i, path in enumerate(paths):  
        current_image = skimage.io.imread(path, as_gray=True, plugin='pil')
        current_image = resize(current_image, size, anti_aliasing=True)
        images.append(current_image)
        stacked = np.stack(images, axis=-1)  
    
    return stacked

def find_mean_img(full_mat, title, f):
    # calculate the average
    result = np.mean(full_mat, axis = 2)
    
    f.imshow(result, cmap='gray')
    f.set_title(f'Mean {title}')
    f.axis('off')
    return result

def find_std_img(full_mat, title, f):
    # calculate the standard deviation
    result = np.std(full_mat, axis = 2)

    f.imshow(result, cmap='gray')
    f.set_title(f'Std {title}')
    f.axis('off')
    return result


# In[ ]:


# run it on our folders
try:
    domain1_images = img2np(domain[0])
    print("done 1")
    domain2_images = img2np(domain[1])
    print("done 2")
    domain3_images = img2np(domain[2])
    print("done 3")
    domain4_images = img2np(domain[5])
    print("done 4")
    domain5_images = img2np(domain[6])
    print("done 5")
except:
    pass


# ## Plot mean

# In[ ]:


fig, axs = plt.subplots(1, 5)
try:
    mean1 = find_mean_img(domain1_images, "CIRRUS", axs[0])
    mean2 = find_mean_img(domain2_images, "PLEX", axs[1])
    mean3 = find_mean_img(domain3_images, "P2C (M)", axs[2])
    mean4 = find_mean_img(domain4_images, "P2C (F)", axs[3])
    mean5 = find_mean_img(domain5_images, "F2C (F)", axs[4])
except:
    pass
fig.tight_layout()
fig.savefig(f"example_results/domain_shift_mean.png", dpi=1200)


# ## Plot absolute mean difference

# In[ ]:


contrast_mean = np.absolute(mean1 - mean2)
# contrast_mean = cv2.absdiff(plex_mean, cirrus_mean)
#contrast_mean = plex_mean * cirrus_mean

plt.imshow(contrast_mean, cmap=variint_map)
plt.title(f'Absolute difference')
plt.axis('off')
plt.tight_layout()
plt.savefig(f"example_results/domain_shift_diff_mean.png", dpi=1200)


# ## Plot standard dev

# In[ ]:


fig, axs = plt.subplots(1, 5)

try:
    std1 = find_std_img(domain1_images, "CIRRUS", axs[0])
    std2 = find_std_img(domain2_images, "PLEX", axs[1])
    std3 = find_std_img(domain3_images, "P2C (M)", axs[2])
    std4 = find_std_img(domain4_images, "P2C (F)", axs[3])
    std5 = find_std_img(domain5_images, "F2C (F)", axs[4])
except:
    pass

fig.tight_layout()
fig.savefig(f"example_results/domain_shift_std.png", dpi=1200)


# In[ ]:





# In[ ]:




