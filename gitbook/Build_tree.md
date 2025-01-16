# 4. Build a Tree
**Now that we've defined all the necessary helper functions, it's time to put everything together and build the decision tree🌳!**

The DecisionTree class is designed to handle both classification and regression tasks. As previouly introduced, a decision tree built using the CART algorithm relies on recursively splitting the dataset into smaller subsets. However, this process cannot be endless. To successfully construct a tree, we need to define stopping criteria (e.g., maximum tree depth, minimum number of samples, or reaching a pure region).

Once the stopping criteria are met, the algorithm must terminate the recursion and assign a prediction to the node, which becomes a leaf node. To achieve this, I created a leaf_node function first within the class. This function ensures that predictions are assigned to nodes where further splitting is not possible, either because the data cannot be split further or the stopping conditions have been reached. 

The fit function orchestrates the training process by invoking the recursive_splitting function, which builds the tree step-by-step. Once the tree is built, it is stored in the self.tree attribute.

The predict function is used to traverse the built tree and generate predictions for new input samples. Starting from the root node, it follows the feature-based split conditions until it reaches a leaf node, where the prediction is returned. This function allows the trained decision tree to be applied effectively to unseen data.