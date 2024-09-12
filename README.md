# Customer Segmentation Using Clustering Model

## Project Overview 
### Context
The Senior Management team from the client, insurance company,  is disagreeing about how customers are being targeted and would like to implement new business startegy. 

Their proposal is to use data and Machine Learning to help segment up their customers based upon their engagement with each of the benefits such as saving plans, loans, wealth management.  This will aid their business understanding of the customer base and enhance the relevancy of targeted messaging and customer communication.

### Actions
To identify meaningful customer segments, we propose applying k-means clustering to the product area data. Here are the key steps:

- **Data Pre-processing:** Feature scaling and dimensionality reduction are necessary to ensure effective clustering.
- **Determine Optimal Clusters:** Utilize the Within Cluster Sum of Squares (WCSS) method to determine the ideal number of clusters.
- **Apply K-Means and Profile Segments:** Apply k-means clustering, append clusters to the customer base, and profile the resulting customer segments to identify differentiating factors.

### Results
Based upon iterative testing using WCSS, 4 clusters looks to be a good customer segmentation.

- For cluster 0, Customers here have balanced credit card usage with low spending and infrequent purchases. However, they pay off their balances regularly, making them low risk but not big spenders.
- For cluster 1,  This group shows the lowest balance, spending, and purchase frequency. They hardly use their credit cards and aren’t contributing much to the bank’s revenue.
- Cluster 3, The high rollers! These customers spend the most, use their cards frequently, and have higher transaction volumes. They’re the key segment for boosting revenue.
- Cluster 4, These customers use their cards often but don’t spend much. They also tend to have lower balances and credit limits, which might be limiting their spending potential.

### Modeling:
#### K-Means

Concept Overview
K-Means is an unsupervised learning algorithm, meaning that it does not look to predict known labels or values, but instead looks to isolate patterns within unlabeled data.

The algorithm works in a way where it partitions data-points into distinct groups (clusters) based upon their similarity to each other.

### Data Preprocessing
There are three vital preprocessing steps for k-means, namely:

- Missing values in the data
- The effect of outliers
- Feature Scaling

#### Missing Values
Missing values can cause issues for k-means, as the algorithm won’t know where to plot those data-points along the dimension where the value is not present.I replaced missing value with a mean of that column. 

```

missing_var = [var for var in creditcard_df.columns if creditcard_df[var].isnull().sum()>0]

creditcard_df["MINIMUM_PAYMENTS"] = creditcard_df["MINIMUM_PAYMENTS"].fillna(creditcard_df["MINIMUM_PAYMENTS"].mean())
creditcard_df["CREDIT_LIMIT"] = creditcard_df["CREDIT_LIMIT"].fillna(creditcard_df["CREDIT_LIMIT"].mean())
```

#### Outliers
As k-means is a distance based algorithm, outliers can cause major problems. I utilized a quantile-based approach to identify and remove outliers from creditcard_df. For each column with float64 or int64 data type, I calculated the 95th and 5th percentiles as thresholds. Then, created a new dataframe with non-outlier data points and calculated the outlier percentage.

```
for i in creditcard_df.select_dtypes(include=['float64','int64']).columns:
  max_thresold = creditcard_df[i].quantile(0.95)
  min_thresold = creditcard_df[i].quantile(0.05)
  creditcard_df_no_outlier = creditcard_df[(creditcard_df[i] < max_thresold) & (creditcard_df[i] > min_thresold)].shape
  print(" outlier in ",i,"is" ,int(((creditcard_df.shape[0]-creditcard_df_no_outlier[0])/creditcard_df.shape[0])*100),"%")
```

#### Feature Scaling
Again, as k-means is a distance based algorithm, in other words it is reliant on an understanding of how similar or different data points are across different dimensions in n-dimensional space, the application of Feature Scaling is extremely important.

Feature Scaling forces the values from different columns to exist on the same scale, in order to enchance the learning capabilities of the model. There are two common approaches for this, Standardization, and Normalization.

I used in-built StandardScaler functionality from scikit-learn to apply Normalization to all of the variables. The reason to create a new object (here called creditcard_scaled_df) is to use the scaled data for clustering, but when profiling the clusters later on, I will want to use the actual percentages as this may make more intuitive business sense. So it’s good to have both options available!

```
# scale the DataFrame
scalar=StandardScaler()
creditcard_scaled_df = scalar.fit_transform(creditcard_df_no_outlier)
```

Dimensionality reduction is a technique used to reduce the number of features in a dataset while retaining as much of the important information as possible.I utilized PCA to reduce dimentiality of the data.

```
pca = PCA(n_components=2)
principal_comp = pca.fit_transform(creditcard_scaled_df)
pca_df = pd.DataFrame(data=principal_comp,columns=["pca1","pca2"])
pca_df.head()
```

#### Finding A Good Value For k
The approach to be utilized here is known as Elbow method which measures the sum of the squared euclidean distances that data points lie from their closest centroid. Elbow method can help to understand the point where adding more clusters provides little extra benefit in terms of separating the data.

By default, the k-means algorithm within scikit-learn will use k = 8, meaning that it will look to split the data into eight distinct clusters. There may a better value that fits our data, and our task!

In the code below I test multiple values for k, and plot how this Elbow method metric changes. 

```
inertia = []
range_val = range(1,15)
for i in range_val:
  kmean = KMeans(n_clusters=i)
  kmean.fit_predict(pd.DataFrame(creditcard_scaled_df))
  inertia.append(kmean.inertia_)
plt.plot(range_val,inertia,'bx-')
plt.xlabel('Values of K') 
plt.ylabel('Inertia') 
plt.title('The Elbow Method using Inertia') 
plt.show()
```
![image](https://github.com/user-attachments/assets/25c3c7d2-543a-45cf-886c-99a86a5cef9e)

Based upon the shape of the above plot - there does appear to be an elbow at k = 4.

#### Model fitting
The below code will instantiate the k-means object using a value for k equal to 4. Then it will fit this k-means object to the scaled dataset to separate the data into fout distinct segments or clusters.

```
kmeans_model=KMeans(4)
kmeans_model.fit_predict(creditcard_scaled_df)
pca_df_kmeans= pd.concat([pca_df,pd.DataFrame({'cluster':kmeans_model.labels_})],axis=1)
```

![image](https://github.com/user-attachments/assets/c908f799-256b-43fa-9144-7ee745031ebf)

Assigned the corresponding cluster labels to each customer in the dataset. This is achieved by creating a new column in the customer data frame and storing the respective cluster name that each customer belongs to.

```
creditcard_cluster_df = pd.concat([creditcard_df,pd.DataFrame({'cluster':kmeans_model.labels_})],axis=1)
```

#### Conclusion
- Cluster 1: This group has excellent payment habits but low credit limits. By increasing their credit limits, we could encourage them to spend more, especially since they’ve shown they can manage their finances well.
- Cluster 2 and 4: These groups have both low spending and low payment activity. To boost their spending, we could introduce targeted promotions with minimum spending requirements or offer special incentives when they increase their credit card usage.
- Cluster 3: This group is already spending the most and using their credit cards frequently. To capitalize on this, we should focus on encouraging even more spending through exclusive promotions or special perks for high spenders. Think along the lines of VIP credit card benefits or rewards for reaching spending milestones.
- 



  
