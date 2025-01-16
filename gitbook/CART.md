# 2. CART(Classification And Regression Tree) for Decision Tree
Decision trees are non-parametric supervised learning techniques, meaning they do not assume any specific distribution for the underlying data. Instead, they learn patterns directly from labeled data, where the target variable is known. One of the most widely used decision tree methodologies is CART (Classification and Regression Tree), first introduced by Breiman et al. in 1984. As its name suggests, CART can be applied to both classification and regression tasks, forming the foundation for many modern decision tree algorithms.

In this project, I will first review the general structure of the CART algorithm and provide a step-by-step guide to implementing it from scratch using only the pandas and numpy libraries in Python. Lastly, I will test my implementation on Python’s built-in datasets — the breast cancer dataset (for classification) and the diabetes dataset (for regression). The performance of my implementation will be compared with the established decision tree model available in the scikit-learn library.
## 2.1 General Structure 
In the image below, I tried to show the general structure of a simple decision tree. In this example, the outcome variable Y is binary, indicating whether a patient has heart disease or not:

- Y = 1: Patient has heart disease.
- Y = 0: Patient does not have heart disease.

![A Simple Decision Tree](dt_general_structure.png)

Key Components of a Decision Tree:
- Root Node: The top node of the tree, representing the initial decision point. All data starts here and is split based on the most important feature.
- Internal Node: Also called decision nodes, these are the points where data is further splitted into subgroups based on certain conditions. Each internal node represents a decision or test on a feature.
- Leaf Node: The final node of the tree, containing the output or decision (e.g., whether the patient has heart disease or not). Leaf nodes do not split further.

## 2.2 Overview of the CART Algorithm Flow
In general, the CART algorithm builds a decision tree by recursively splitting the dataset into subsets based on feature values. The goal is to create pure subsets where the target variable is as homogeneous as possible. Below are the key steps in the CART algorithm.
1. Choose the Best Split: Search through all the predictors and possible split points (thresholds). Pick the predctor and split point that result in the best split, minimizing impurity (e.g., Gini index for classification or Mean Squared Error for regression). This feature and threshold combination become the root node.
2. Split the Data: Divides the data into two subsets based on the chosen feature and threshold:
    - One subset where the condition is true (e.g., feature value ≤ threshold).
    - Another subset where the condition is false (e.g., feature value > threshold).
3. Recursive binary splitting: Starting from the previously formed region. Repeat step 1 and 2, and stop once a stopping criterion is reached. 
4. Assign Outcomes to Leaf Nodes: Once the stopping criterion is reached and no further splits are possible, assign a final prediction to each leaf node:
    - For classification tasks, assign the majority class of the samples in the node.
    - For regression tasks, assign the mean target value of the samples in the node.
