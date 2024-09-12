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
  
  For cluster 0, Customers here have balanced credit card usage with low spending and infrequent purchases. However, they pay off their balances regularly, making them low risk but not big spenders.
  For cluster 1,  This group shows the lowest balance, spending, and purchase frequency. They hardly use their credit cards and aren’t contributing much to the bank’s revenue.
  Cluster 3, The high rollers! These customers spend the most, use their cards frequently, and have higher transaction volumes. They’re the key segment for boosting revenue.
  Cluster 4, These customers use their cards often but don’t spend much. They also tend to have lower balances and credit limits, which might be limiting their spending potential.

  
