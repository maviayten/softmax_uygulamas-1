#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


iris = datasets.load_iris()
iris_df = pd.DataFrame(data=iris.data,columns=iris.feature_names)
iris_df['target'] = iris.target
iris_df.head()


# In[4]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.pairplot(iris_df,hue='target',markers=["o","s","D"])
plt.suptitle("iris veri seti çift değişkenli saçılım grafikleri",y=1.02)
plt.show()


# In[9]:


import seaborn as sns
import matplotlib.pyplot as plt


plt.figure(figsize=(12, 8))


plt.subplot(2, 2, 1)
sns.boxplot(x="target", y="sepal length (cm)", data=iris_df)
plt.title("Sepal Length (cm) - Kutu Grafiği")

plt.subplot(2, 2, 2)
sns.boxplot(x="target", y="sepal width (cm)", data=iris_df)
plt.title("Sepal Width (cm) - Kutu Grafiği")


plt.subplot(2, 2, 3)
sns.boxplot(x="target", y="petal length (cm)", data=iris_df)
plt.title("Petal Length (cm) - Kutu Grafiği")

plt.subplot(2, 2, 4)
sns.boxplot(x="target", y="petal width (cm)", data=iris_df)
plt.title("Petal Width (cm) - Kutu Grafiği")


plt.tight_layout()
plt.show()


# In[11]:


import seaborn as sns
import matplotlib.pyplot as plt


plt.figure(figsize=(12, 8))


plt.subplot(2, 2, 1)
sns.histplot(data=iris_df, x="sepal length (cm)", hue="target", kde=True, palette={0: "lightcoral", 1: "lightskyblue", 2: "mediumseagreen"})
plt.title("Sepal Length (cm) - Histogram")


plt.subplot(2, 2, 2)
sns.histplot(data=iris_df, x="sepal width (cm)", hue="target", kde=True, palette={0: "lightcoral", 1: "lightskyblue", 2: "mediumseagreen"})
plt.title("Sepal Width (cm) - Histogram")


plt.subplot(2, 2, 3)
sns.histplot(data=iris_df, x="petal length (cm)", hue="target", kde=True, palette={0: "lightcoral", 1: "lightskyblue", 2: "mediumseagreen"})
plt.title("Petal Length (cm) - Histogram")


plt.subplot(2, 2, 4)
sns.histplot(data=iris_df, x="petal width (cm)", hue="target", kde=True, palette={0: "lightcoral", 1: "lightskyblue", 2: "mediumseagreen"})
plt.title("Petal Width (cm) - Histogram")


plt.tight_layout()
plt.show()


# In[12]:


import seaborn as sns
import matplotlib.pyplot as plt


correlation_matrix = iris_df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", square=True, cbar_kws={'shrink': .8})
plt.title("Iris Veri Seti Korelasyon Matrisi Isı Haritası")
plt.show()


# In[13]:


basic_stats=iris_df.describe()
basic_stats


# In[14]:


data_types=iris_df.dtypes
data_types


# In[16]:


from sklearn.preprocessing import StandardScaler
import numpy as np

features =iris_df.drop('target',axis=1)
target=iris_df['target']

plt.figure(figsize=(12,8))
plt.suptitle('özellikler için kutu grafiği',fontsize=16)
sns.boxplot(data=features,orient='h',palette='Set2')
plt.show()


# In[17]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
normalized_data = scaler.fit_transform(iris_df.iloc[:, :-1]) 

normalized_df = pd.DataFrame(normalized_data, columns=iris_df.columns[:-1])
normalized_df["target"] = iris_df["target"]


normalized_df.head()


# In[18]:


from sklearn.model_selection import train_test_split

X = normalized_df.drop("target", axis=1) 
y = normalized_df["target"]             

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("Eğitim seti boyutu:", X_train.shape)
print("Test seti boyutu:", X_test.shape)


# In[19]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


softmax= LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42)


softmax.fit(X_train, y_train)

y_pred = softmax.predict(X_test)


accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')


conf_matrix = confusion_matrix(y_test, y_pred)

print("Karmaşıklık Matrisi\n",conf_matrix)
print("Doğruluk:", accuracy)
print("Kesinlik",precision)
print("Duyarlılık", recall)
print("F1-Skor",f1)


# In[20]:


from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

y_bin = label_binarize(y, classes=[0, 1, 2])


clf = OneVsRestClassifier(LogisticRegression(solver='lbfgs', random_state=42))
y_score = clf.fit(X_train, y_bin[y_train.index]).decision_function(X_test)

fpr, tpr, roc_auc = {}, {}, {}
for i in range(y_bin.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_bin[y_test.index][:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])


plt.figure(figsize=(8, 6))
colors = ['blue', 'red', 'green']
for i, color in zip(range(y_bin.shape[1]), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC curve of class {i} (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Multi-class ROC Curve')
plt.legend(loc="lower right")
plt.show()


# In[21]:


coefficients=softmax.coef_
intercepts=softmax.intercept_
print("model katsayıları(coef_):")
print(coefficients)
print("\nModel Kesişim Noktaları (Intercept_):")
print(intercepts)


# In[ ]:




