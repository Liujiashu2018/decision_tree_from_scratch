# 4. Build a Tree
**Now that we've defined all the necessary helper functions, it's time to put everything together and build the decision tree🌳!**

The DecisionTree class is designed to handle both classification and regression tasks. As previouly introduced, a decision tree built using the CART algorithm relies on recursively splitting the dataset into smaller subsets. However, this process cannot be endless. To successfully construct a tree, we need to define stopping criteria (e.g., maximum tree depth, minimum number of samples, or reaching a pure region).

Once the stopping criteria are met, the algorithm must terminate the recursion and assign a prediction to the node, which becomes a leaf node. To achieve this, I created a leaf_node function first within the class. This function ensures that predictions are assigned to nodes where further splitting is not possible, either because the data cannot be split further or the stopping conditions have been reached. 

The fit function orchestrates the training process by invoking the recursive_splitting function, which builds the tree step-by-step. Once the tree is built, it is stored in the self.tree attribute.

The predict function is used to traverse the built tree and generate predictions for new input samples. Starting from the root node, it follows the feature-based split conditions until it reaches a leaf node, where the prediction is returned. This function allows the trained decision tree to be applied effectively to unseen data.

```python
# Build a tree
class DecisionTree:
    def __init__(self, task = 'classification', criterion = 'gini', min_sample=2, max_depth=None):
        self.task = task 
        self.criterion = criterion
        self.min_sample = min_sample
        self.max_depth = max_depth
        self.tree = None

    def leaf_node(self, data, target_index):
        target_val = [row[target_index] for row in data]
        if self.task == 'classification':
             # Find the most frequent element (class label) in the list
            return max(set(target_val), key = target_val.count)
        else:
            return np.mean(target_val)
    
    def recursive_splitting(self, data, target_index, depth = 0):
        # Define a stopping criteria 
        if depth >= self.max_depth or len(data)< self.min_sample:
            return self.leaf_node(data, target_index)

        # Use the previously defined function to find the best split
        best_feature_index, best_threshold, best_score = find_best_split(data, target_index)
        if best_feature_index is None:
            return self.leaf_node(data, target_index)

        # Use the previously defined function to split the data
        left, right = split_data (best_feature_index, best_threshold, data)
        if not left or not right: # empty groups
            return self.leaf_node(data, target_index)

        # Initiate the recursive process to build the right and left trees respectively 
        left_tree = self.recursive_splitting(left, target_index, depth + 1)
        right_tree = self.recursive_splitting(right, target_index, depth + 1)

        return {
            'feature': best_feature_index,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree
        }
    
    def fit(self, data, target_index):
        self.tree = self.recursive_splitting(data, target_index)
    
    def predict(self, row):
        '''
        While not a leaf, evaluate the feature against the threshold 
        and update the node either to the left or right subtree.
        '''
        node = self.tree # initiate a root node
        
        # Check whether the node is a terminal node. Terminal nodes will be stored in as dictionaries.
        # In contrast, if the node is a leaf node, it will not be a dictionary but a single value like a class label
        # for classification or a numerical value a numeric prediction for regression. In that case, the loop stops.  
        while isinstance(node, dict):  
            if row[node['feature']] <= node['threshold']:
                node = node['left']
            else:
                node = node['right']
        return node
```