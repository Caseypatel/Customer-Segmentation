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

#### Outliers
As k-means is a distance based algorithm, outliers can cause major problems. I utilized a quantile-based approach to identify and remove outliers from creditcard_df. For each column with float64 or int64 data type, I calculated the 95th and 5th percentiles as thresholds. Then, created a new dataframe with non-outlier data points and calculated the outlier percentage.

#### Feature Scaling




  
