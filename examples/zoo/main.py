import argparse
import os.path as osp

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from abl.bridge import SimpleBridge
from abl.data.evaluation import ReasoningMetric, SymbolAccuracy
from abl.learning import ABLModel
from abl.reasoning import Reasoner
from abl.utils import ABLLogger, confidence_dist, print_log

from get_dataset import load_and_preprocess_dataset, split_dataset
from kb import ZooKB


def transform_tab_data(X, y):
    return ([[x] for x in X], [[y_item] for y_item in y], [0] * len(y))

def consitency(data_example, candidates, candidate_idxs, reasoning_results):
    pred_prob = data_example.pred_prob
    model_scores = confidence_dist(pred_prob, candidate_idxs)
    rule_scores = np.array(reasoning_results)
    scores = model_scores + rule_scores
    return scores

def main():
    parser = argparse.ArgumentParser(description="Zoo example")
    parser.add_argument(
        "--loops", type=int, default=3, help="number of loop iterations (default : 3)"
    )
    args = parser.parse_args()
    
    # Build logger
    print_log("Abductive Learning on the ZOO example.", logger="current")

    ### Working with Data
    print_log("Working with Data.", logger="current")
    
    X, y = load_and_preprocess_dataset(dataset_id=62)
    X_label, y_label, X_unlabel, y_unlabel, X_test, y_test = split_dataset(X, y, test_size=0.3)
    label_data = transform_tab_data(X_label, y_label)
    test_data = transform_tab_data(X_test, y_test)
    train_data = transform_tab_data(X_unlabel, y_unlabel)

    ### Building the Learning Part
    print_log("Building the Learning Part.", logger="current")
    
    # Build base model
    base_model = RandomForestClassifier()

    # Build ABLModel
    model = ABLModel(base_model)

    ### Building the Reasoning Part
    print_log("Building the Reasoning Part.", logger="current")
    
    # Build knowledge base
    kb = ZooKB()
    
    # Create reasoner
    reasoner = Reasoner(kb, dist_func=consitency)

    ### Building Evaluation Metrics
    print_log("Building Evaluation Metrics.", logger="current")
    metric_list = [SymbolAccuracy(prefix="zoo"), ReasoningMetric(kb=kb, prefix="zoo")]
    
    ### Bridging learning and reasoning
    print_log("Bridge Learning and Reasoning.", logger="current")
    bridge = SimpleBridge(model, reasoner, metric_list)
    
    # Retrieve the directory of the Log file and define the directory for saving the model weights.
    log_dir = ABLLogger.get_current_instance().log_dir
    weights_dir = osp.join(log_dir, "weights")
    
    # Performing training and testing
    print_log("------- Use labeled data to pretrain the model -----------", logger="current")
    base_model.fit(X_label, y_label)
    print_log("------- Test the initial model -----------", logger="current")
    bridge.test(test_data)
    print_log("------- Use ABL to train the model -----------", logger="current")
    bridge.train(train_data=train_data, label_data=label_data, loops=args.loops, segment_size=len(X_unlabel), save_dir=weights_dir)
    print_log("------- Test the final model -----------", logger="current")
    bridge.test(test_data)


if __name__ == "__main__":
    main()
