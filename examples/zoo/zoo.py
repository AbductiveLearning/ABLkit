import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from z3 import Solver, Int, If, Not, Implies, Sum, sat
import openml
from abl.learning import ABLModel
from abl.reasoning import KBBase, Reasoner
from abl.evaluation import ReasoningMetric, SymbolMetric
from abl.bridge import SimpleBridge
from abl.utils.utils import confidence_dist

class ZooKB(KBBase):
    def __init__(self):
        super().__init__(pseudo_label_list=list(range(7)), use_cache=False)
        
        # Use z3 solver 
        self.solver = Solver()

        # Load information of Zoo dataset
        dataset = openml.datasets.get_dataset(dataset_id = 62, download_data=False, download_qualities=False, download_features_meta_data=False)
        X, y, categorical_indicator, attribute_names = dataset.get_data(target=dataset.default_target_attribute)
        self.attribute_names = attribute_names
        self.target_names = y.cat.categories.tolist()
        print("Attribute names are: ", self.attribute_names)
        print("Target names are: ", self.target_names)
        # self.attribute_names = ["hair", "feathers", "eggs", "milk", "airborne", "aquatic", "predator", "toothed", "backbone", "breathes", "venomous", "fins", "legs", "tail", "domestic", "catsize"]
        # self.target_names = ["mammal", "bird", "reptile", "fish", "amphibian", "insect", "invertebrate"]

        # Define variables
        for name in self.attribute_names+self.target_names:
            exec(f"globals()['{name}'] = Int('{name}')") ## or use dict to create var and modify rules
        # Define rules
        rules = [
            Implies(milk == 1, mammal == 1),
            Implies(mammal == 1, milk == 1),
            Implies(mammal == 1, backbone == 1),
            Implies(mammal == 1, breathes == 1),
            Implies(feathers == 1, bird == 1),
            Implies(bird == 1, feathers == 1),
            Implies(bird == 1, eggs == 1),
            Implies(bird == 1, backbone == 1),
            Implies(bird == 1, breathes == 1),
            Implies(bird == 1, legs == 2),
            Implies(bird == 1, tail == 1),
            Implies(reptile == 1, backbone == 1),
            Implies(reptile == 1, breathes == 1),
            Implies(reptile == 1, tail == 1),
            Implies(fish == 1, aquatic == 1),
            Implies(fish == 1, toothed == 1),
            Implies(fish == 1, backbone == 1),
            Implies(fish == 1, Not(breathes == 1)),
            Implies(fish == 1, fins == 1),
            Implies(fish == 1, legs == 0),
            Implies(fish == 1, tail == 1),
            Implies(amphibian == 1, eggs == 1),
            Implies(amphibian == 1, aquatic == 1),
            Implies(amphibian == 1, backbone == 1),
            Implies(amphibian == 1, breathes == 1),
            Implies(amphibian == 1, legs == 4),
            Implies(insect == 1, eggs == 1),
            Implies(insect == 1, Not(backbone == 1)),
            Implies(insect == 1, legs == 6),
            Implies(invertebrate == 1, Not(backbone == 1))
        ]
        # Define weights and sum of violated weights
        self.weights = {rule: 1 for rule in rules}
        self.total_violation_weight = Sum([If(Not(rule), self.weights[rule], 0) for rule in self.weights])
        
    def logic_forward(self, pseudo_label, data_point):
        attribute_names, target_names = self.attribute_names, self.target_names
        solver = self.solver
        total_violation_weight = self.total_violation_weight
        pseudo_label, data_point = pseudo_label[0], data_point[0]
        
        self.solver.reset()
        for name, value in zip(attribute_names, data_point):
            solver.add(eval(f"{name} == {value}"))
        for cate, name in zip(self.pseudo_label_list,target_names):
            value = 1 if (cate == pseudo_label) else 0
            solver.add(eval(f"{name} == {value}"))
            
        if solver.check() == sat:
            model = solver.model()
            total_weight = model.evaluate(total_violation_weight)
            # violated_rules = [str(rule) for rule in self.weights if model.evaluate(Not(rule))]
            # print("Total violation weight for the given data point:", total_weight)
            # print("Violated rules:", violated_rules)
            return total_weight.as_long()
        else:
            # No solution found
            return 1e10

def consitency(data_sample, candidates, candidate_idxs, reasoning_results):
    pred_prob = data_sample.pred_prob
    model_scores = confidence_dist(pred_prob, candidate_idxs)
    rule_scores = np.array(reasoning_results)
    scores = model_scores + rule_scores
    return scores

# Function to load and preprocess the dataset
def load_and_preprocess_dataset(dataset_id):
    dataset = openml.datasets.get_dataset(dataset_id, download_data=True, download_qualities=False, download_features_meta_data=False)
    X, y, _, attribute_names = dataset.get_data(target=dataset.default_target_attribute)
    # Convert data types
    for col in X.select_dtypes(include='bool').columns:
        X[col] = X[col].astype(int)
    y = y.cat.codes.astype(int)
    X, y = X.to_numpy(), y.to_numpy()
    return X, y

# Function to split data (one shot)
def split_dataset(X, y, test_size = 0.3):
    # For every class: 1 : (1-test_size)*(len-1) : test_size*(len-1)
    label_indices, unlabel_indices, test_indices = [], [], []
    for class_label in np.unique(y):
        idxs = np.where(y == class_label)[0]
        np.random.shuffle(idxs)
        n_train_unlabel = int((1-test_size)*(len(idxs)-1))
        label_indices.append(idxs[0])
        unlabel_indices.extend(idxs[1:1+n_train_unlabel])
        test_indices.extend(idxs[1+n_train_unlabel:])
    X_label, y_label = X[label_indices], y[label_indices]
    X_unlabel, y_unlabel = X[unlabel_indices], y[unlabel_indices]
    X_test, y_test = X[test_indices], y[test_indices]
    return X_label, y_label, X_unlabel, y_unlabel, X_test, y_test

    
if __name__ == "__main__":
    '''
    Working with data
    '''
    # Load and preprocess the Zoo dataset
    X, y = load_and_preprocess_dataset(dataset_id=62)
    print("Shape of X and y:", X.shape, y.shape)
    print("First five elements of X:")
    print(X[:5])
    print("First five elements of y:")
    print(y[:5])
    
    # Split data into labeled/unlabeled/test data
    X_label, y_label, X_unlabel, y_unlabel, X_test, y_test = split_dataset(X, y, test_size=0.3)
    
    # Transform tabluar data to the format required by ABL, which is a tuple of (X, ground truth of X, reasoning results)
    # For tabular data in abl, each sample contains a single instance (a row from the dataset).
    # For these tabular data samples, the reasoning results are expected to be 0, indicating no rules are violated.
    def transform_tab_data(X, y):
        return ([[x] for x in X], [[y_item] for y_item in y], [0] * len(y))
    label_data = transform_tab_data(X_label, y_label)
    test_data = transform_tab_data(X_test, y_test)
    train_data = transform_tab_data(X_unlabel, y_unlabel)
    
    '''
    Building the learning part
    '''
    rf = RandomForestClassifier()
    # Pre-train the machine learning model
    rf.fit(X_label, y_label)
    # # Test the initial model
    # y_test_pred = rf.predict(X_test)
    # labeled_test_acc = accuracy_score(y_test, y_test_pred)
    # print(labeled_test_acc)
    model = ABLModel(rf)
    
    '''
    Building the reasoning part
    '''
    # Create the knowledge base for Zoo
    kb = ZooKB()
    # # Test ZooKB
    # pseudo_label = [0]
    # data_point = [np.array([1,0,0,1,0,0,1,1,1,1,0,0,4,0,0,1,1])]
    # print(kb.logic_forward(pseudo_label, data_point))
    # for x,y_item in zip(X, y):
    #     print(x,y_item)
    #     print(kb.logic_forward([y_item], [x]))
    reasoner = Reasoner(kb, dist_func=consitency)
    
    '''
    Building evaluation metrics
    '''
    metric_list = [SymbolMetric(prefix="zoo"), ReasoningMetric(kb=kb, prefix="zoo")]
    
    '''
    Bridging Learning and Reasoning
    '''
    bridge = SimpleBridge(model, reasoner, metric_list)
    
    # Test the initial model
    print("------- Test the initial model -----------")
    bridge.test(test_data)
    print("------- Use ABL to train the model -----------")
    # Use ABL to train the model
    bridge.train(train_data=train_data, label_data=label_data, loops=3, segment_size=len(X_unlabel))
    print("------- Test the final model -----------")
    # Test the final model
    bridge.test(test_data)
    