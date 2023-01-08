import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, silhouette_score, silhouette_samples

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

def main():
    #Reading Dataset

    df = pd.read_csv('football.csv', encoding='latin-1')

    kf = df.copy(deep=True)

    # Preprocessing Data set
    df.drop(['First_Name'], axis=1, inplace=True)

    df.drop(['Last_Name'], axis=1, inplace=True)

    # print(df.shape)

    # df['total_skills'] = df['wontackles'] + df['wonduels'] + df['wonpasses']

    df['wontackles']=np.around(df['wontackles'],decimals=5)
    df['wonduels'] = np.around(df['wonduels'], decimals=5)
    df['wonpasses']= np.around(df['wonpasses'], decimals=-5)

    flage = 100000
    arr1=[]
    arr2=[]
    arr3=[]
    df['wontackles'] *= flage
    df['wonduels'] *= flage
    df['wonpasses'] *= flage


    # pd.to_numeric(df['wontackles'], downcast='integer')
    # pd.to_numeric(df['wonduels'], downcast='integer')
    # pd.to_numeric(df['wonpasses'], downcast='integer')
    df['wontackles']=df['wontackles'].astype(str)
    df['wonduels']=df['wonduels'].astype(str)
    df['wonpasses']=df['wonpasses'].astype(str)

    for i in df['wontackles']:
         arr1.append(i.split('.')[0])
    for i in df['wonduels']:
        arr2.append(i.split('.')[0])
    for i in df['wonpasses']:
        arr3.append(i.split('.')[0])

    a=np.array(arr1)
    b=np.array(arr2)
    c=np.array(arr3)

    df['wontackles'] =a
    df['wonduels']=b
    df['wonpasses']=c

    df['total_skills']=df['wontackles']+df['wonduels']+df['wonpasses']

    df['total_skills']=df['total_skills'].astype(int)

    df['total_skills']/=(flage*flage*10)

    df.drop(['wontackles', 'wonduels', 'wonpasses'], axis=1, inplace=True)

    # create new plot and data
    plt.plot()
    X = np.array(list(zip(df['Player Id'], df['total_skills']))).reshape(len(df['Player Id']), 2)
    colors = ['b', 'g', 'r']
    markers = ['o', 'v', 's']

    # k means determine k
    distortions = []
    K = range(1, 10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])

    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()


    kmeans = KMeans(n_clusters=4)

    KModel = kmeans.fit(df)

    y_kmeans = kmeans.predict(df)

    plt.scatter(df['Player Id'], df['total_skills'], c=y_kmeans, s=50, cmap='viridis')

    centers = kmeans.cluster_centers_

    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

    plt.show()

    # Finding Player Name

    df['cluster'] = kmeans.labels_

    flag = df.iloc[265]['total_skills']


    arr = []

    temp = []

    n = 0

    for i in range(0, len(df.index)):
        if df['cluster'][i] == 0:
            arr.append(df['total_skills'][i])
            n += 1

    for i in range(0, n):
        temp.append(abs(arr[i] - flag))

    a2 = np.array(temp)

    minindx = a2.argmin()
    

    for i in range(0,len(df.index)):
        if df['total_skills'][i]==arr[minindx]:
            name=kf['First_Name'][i]

    # Print the subtitution

    print("Substitute for the player is "+str(name))

    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(df)

    silhouette_avg = silhouette_score(X, kmeans.labels_)
    print("For n_clusters =",str(4),
          "The average silhouette_score is :", silhouette_avg)
main()
