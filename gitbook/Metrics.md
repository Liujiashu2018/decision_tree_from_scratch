# 3. Metrics to Decide a Good Split
The CART algorithm evaluates possible splits to identify the one that minimizes impurity or error. Different metrics are used depending on the task. 
## 3.1 Gini Impurity
Suppose the response variable $Y$ has $C$ different classes, and the tree has $R$ total regions/nodes. We want to first measure the purity of a single node.  

Let $p$ = proportion of cases in region/node $r$ that are of class $c$,  the node/region's Gini Impurity, or Gini Index, is calculated as:  

$$
G_r = \sum_{c=1}^{C} \left( p \cdot (1 - p) \right)
$$

The smaller $G$ is, the purer the region $r$ becomes. When $G = 0$, the region is perfectly pure, meaning all samples in the node belong to a single class.

To determine the "best" binary split, we compute the **weighted Gini** Impurity as follows:
$$
G_{\text{weighted}} = \sum_{r=1}^{R} \left( G_r \cdot \frac{\text{\# cases in region } r}{\text{total \# cases}} \right)
$$
The split with the lowest $G_{\text{weighted}}$ is considered the "best" split. This metric is primarily used in **classification tasks**.

## 3.2  Entropy
Entropy is another popular metric for **classification tasks**, measuring the impurity of a region. Similar to Gini Impurity, lower entropy indicates more homogeneity in a region. 
Entropy is calculated as: 
$$
Entropy_r = -\sum_{c=1}^{C} \left( p \cdot log_2(p) \right)
$$
where
- Entropy_r is the entropy of a region/node $r$.
- C is the total number of classes in the response vairable $Y$.
- $p$ is the proportion of cases in region/node $r$ that beglong to class $c$.
To determine the "best" binary split using entropy, we compute the **weighted entropy** as follows:
$$
Entropy_{\text{weighted}} = \sum_{r=1}^{R} \left( Entropy_r \cdot \frac{\text{\# cases in region } r}{\text{total \# cases}} \right)
$$

## 3.3 Mean Squared Error (MSE)
For **regression tasks**, we use the Mean Squared Error (MSE) as the metric to measure the impurity of a region. MSE quantifies the variance within a region by calculating the average squared difference between the actual values and the mean value of the region. It is defined as:
$$
MSE_r = \frac{1}{N_r} \sum_{i=1}^{N_r} \left( y_i - \bar{y_r} \right)^2
$$
where:
- $MSE_r$ is the Mean Squared Error for region/node $r$.
- $N_r$ is the number of cases in region $r$.
- $y_i$ represents the true target value of the $i$-th case in region $r$.
- $\bar{y_r}$ is the mean target value of all cases in region $r$.

The smaller $MSE_r$ is, the less variance there is in the region, indicating a better split. If $MSE_r = 0$, the region is perfectly pure.

To determine the "best" binary split using MSE, we compute the **weighted MSE** as follows:
$$
MSE_{\text{weighted}} = \sum_{r=1}^{R} \left( MSE_r \cdot \frac{\text{\# cases in region } r}{\text{total \# cases}} \right)
$$

## 3.4 Finding the best split
After defining functions to compute the metrics for classification and regression tasks, we can create a new function to identify the best split. My find_best_split function incorporates four key elements:
1. The input dataset.
2. The index of the target variable column.
3. A task parameter that specifies whether the task is classification or regression.
4. The metric used to calculate impurity and evaluate splits (e.g., Gini, Entropy, or MSE).

This function iterates through all features and possible split thresholds, evaluates the impurity for each split, and selects the feature and threshold combination that minimizes the chosen metric. The result is the optimal split for the current dataset.