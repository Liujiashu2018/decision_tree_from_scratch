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

```python
# Define Helper Functions
# Split a dataset based on a feature and a given threshold
def split_data (feature_index, threshold, data):
    left, right = [], []
    for row in data:
        if row[feature_index] <= threshold:
            left.append(row)
        else:
            right.append(row)
    #left = np.array(left)
    #right = np.array(right)
    return left, right

# Gini Impurity 
# weighted gini impurity = (left group size / total samples)* Gini_left + (right group size / total samples)* Gini_right
def gini_impurity(groups, classes, class_index):
    '''
    Parameters:
        groups: A list of all subsets (left and right groups)
        classes: Target classes, e.g. 0 & 1, male and female
        class_index: Column index of class label in the data
    '''
    total_sample = sum(len(group) for group in groups) 
    weighted_gini = 0
    for group in groups:
        group_size = len(group)
        if group_size == 0:
            continue # skip empty groups
        
        gini = 0
        for class_label in classes:
            class_size = sum(1 for row in group if row[class_index] == class_label)
            p = class_size/group_size 
            gini += p**2
        group_gini = 1-gini
        weighted_gini += group_gini * (group_size/total_sample)

    return weighted_gini
```

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

``` python
# Entropy
def entropy(groups, classes, class_index):
    total_sample = sum(len(group) for group in groups) 
    weighted_entropy = 0
    for group in groups:
        group_size = len(group)
        if group_size == 0:
            continue # skip empty groups
    
        entropy = 0
        for class_label in classes:
            class_size = sum(1 for row in group if row[class_index] == class_label)
            p = class_size/group_size
            if p > 0:
                entropy -= p * np.log2(p)
        weighted_entropy += entropy * (group_size/total_sample)
    
    return weighted_entropy
```

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

```python
# MSE for regression task
def mse(groups, target_index):
    total_sample = sum(len(group) for group in groups)
    weighted_mse = 0
    
    for group in groups:
        group_size = len(group)
        if group_size == 0:
            continue # skip empty groups

        group_value = [row[target_index] for row in group]
        group_mean = np.mean(group_value)
        group_mse = sum((x - group_mean) ** 2 for x in group_value) / group_size
        weighted_mse += group_mse * (group_size/total_sample)

    return weighted_mse
```

## 3.4 Finding the best split
After defining functions to compute the metrics for classification and regression tasks, we can create a new function to identify the best split. My find_best_split function incorporates four key elements:
1. The input dataset.
2. The index of the target variable column.
3. A task parameter that specifies whether the task is classification or regression.
4. The metric used to calculate impurity and evaluate splits (e.g., Gini, Entropy, or MSE).

``` python
# Find the best split
def find_best_split(data, target_index, task = 'classification', criterion = 'gini'):
    '''
    Parameters:
        data: Input dataset where rows are samples and columns are features.
        target_index: The index of the column which is chosen to be the target variable.
        task: Classification or Regression. (default is classification)
        criterion: Gini, Entropy, or MSE (default criterion is gini index)
    
    Returns:
        best_feature_index: The index of the best feature of splitting.
        best_threshold: The threshold or values that gives the best split.
        best_score: The best (lowest) gini impurity score for the split.
    '''
    best_feature_index = None
    best_threshold = None
    best_score = float('inf')

    data = np.array(data)
    n_features = data.shape[1] # number of columns in the data

    # Extract unique class labels from the target column
    if task == 'classification':
        classes = np.unique(data[:, target_index])

    for feature_index in range(n_features):
        if feature_index == target_index:
            continue # skip the target column

        # Extract unique values for each feature to test as splitting thresholds:
        thresholds = np.unique(data[:, feature_index])
        for threshold in thresholds:
            # Split the dataset into left and right groups using the defined function
            left_group, right_group = split_data(feature_index, threshold, data)
            groups = [left_group, right_group]

            if len(left_group) == 0 or len(right_group) == 0:
                continue # skip empty groups 
            
            if task == 'classification':
                if criterion == 'gini':
                    score = gini_impurity(groups, classes, target_index)
                elif criterion == 'entropy':
                    score = entropy(groups, classes, target_index)
            else:
                score = mse(groups, target_index)

            # Update the best split if the current score is better (lower)
            if score < best_score:
                best_score = score
                best_feature_index = feature_index
                best_threshold = threshold
                
    return best_feature_index, best_threshold, best_score
```
This function iterates through all features and possible split thresholds, evaluates the impurity for each split, and selects the feature and threshold combination that minimizes the chosen metric. The result is the optimal split for the current dataset.