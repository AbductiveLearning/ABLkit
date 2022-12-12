import numpy as np
from utils.utils import flatten, reform_idx


def get_rules_from_data(equations_true):
    SAMPLES_PER_RULE = 3

    select_index = np.random.randint(len(equations_true), size=SAMPLES_PER_RULE)
    select_equations = np.array(equations_true)[select_index]


def get_consist_idx(exs, abducer):
    consistent_ex_idx = []
    label = []
    for idx, e in enumerate(exs):
        if abducer.kb.logic_forward([e]):
            consistent_ex_idx.append(idx)
            label.append(e)
    return consistent_ex_idx, label

def get_label(exs, solution, abducer):
    all_address_flag = reform_idx(solution, exs)
    consistent_ex_idx = []
    label = []
    for idx, ex in enumerate(exs):
        address_idx = [i for i, flag in enumerate(all_address_flag[idx]) if flag != 0]
        candidate = abducer.kb.address_by_idx([ex], 1, address_idx, True)
        if len(candidate) > 0:
            consistent_ex_idx.append(idx)
            label.append(candidate[0][0])
    return consistent_ex_idx, label


def get_percentage_precision(select_X, consistent_ex_idx, equation_label):
    
    images = []
    for idx in consistent_ex_idx:
        images.append(select_X[idx])
        
    ## TODO
    model_labels = model.predict(images)
    
    assert(len(flatten(model_labels)) == len(flatten(equation_label)))
    return (flatten(model_labels) == flatten(equation_label)).sum() / len(flatten(model_labels))
    
    
    
    

def abduce_and_train(model, abducer, train_X_true, select_num):

    select_index = np.random.randint(len(train_X_true), size=select_num)
    select_X = train_X_true[select_index]

    
    
    exs = select_X.predict()
    # e.g. when select_num == 10, exs = [[1, '+', 0, '=', 1, 0], [1, '+', 0, '=', 1, 0], [1, '+', 0, '=', 1, 0], [0, '+', 0, '=', 0], [1, '+', 0, '=', 1, 0],\
    #                                    [1, '+', 0, '=', 1, 0], [1, '+', 0, '=', 1, 0], [1, '+', 0, '=', 1, 0], [0, '+', 0, '=', 0], [1, '+', 0, '=', 1, 0]]

    print("This is the model's current label:", exs)

    # 1. Check if it can abduce rules without changing any labels
    consistent_ex_idx, equation_label = get_consist_idx(exs)

    
    max_abduce_num = 10
    if len(consistent_ex_idx) == 0:

        # 2. Find the possible wrong position in symbols and Abduce the right symbol through logic module
        solution = abducer.zoopt_get_solution(exs, [1] * len(exs), max_abduce_num)
        consistent_ex_idx, equation_label = get_label(exs, solution, abducer)
        
        # Still cannot find
        if len(consistent_ex_idx) == 0:
            return 0, 0

    
    ## TODO: train
    # train_pool_X = np.concatenate(select_X[consistent_ex_idx]).reshape(
    #     -1, h, w, d)
    # train_pool_Y = np_utils.to_categorical(
    #     flatten(exs[consistent_ex_idx]),
    #     num_classes=len(labels))  # Convert the symbol to network output
    # assert (len(train_pool_X) == len(train_pool_Y))
    # print("\nTrain pool size is :", len(train_pool_X))
    # print("Training...")
    # base_model.fit(train_pool_X,
    #                 train_pool_Y,
    #                 batch_size=BATCHSIZE,
    #                 epochs=NN_EPOCHS,
    #                 verbose=0)

    # consistent_percentage, batch_label_model_precision = get_percentage_precision(
    #     base_model, select_equations, consist_re, shape)

    consistent_percentage = len(consistent_ex_idx) / select_num
    batch_label_model_precision = get_percentage_precision(exs, consistent_ex_idx, equation_label)

    return consistent_percentage, batch_label_model_precision

def get_rules(exs):
    consistent_ex_idx, equation_label = get_consist_idx(exs)
    consist_exs = []
    for idx in consistent_ex_idx:
        consist_exs.append(exs[idx])
    if len(consist_exs) == 0:
        return None
    else:
        return abducer.abduce_rule(consist_exs)



def get_rules_from_data(train_X_true, samples_per_rule, logic_output_dim):
    rules = []
    for _ in range(logic_output_dim):
        while True:
            select_index = np.random.randint(len(train_X_true), size=samples_per_rule)
            select_X = train_X_true[select_index]
            
            ## TODO
            exs = select_X.predict()
            rule = get_rules(exs)
            if rule != None:
                break
        rules.append(rule)
    return rules


def get_mlp_vector(X, rules):
    
    ## TODO
    exs = np.argmax(model.predict(X))
    
    vector = []
    for rule in rules:
        if abducer.kb.consist_rule(exs, rule):
            vector.append(1)
        else:
            vector.append(0)
    return vector

def get_mlp_data(X_true, X_false, rules):
    mlp_vectors = []
    mlp_labels = []
    for X in X_true:
        mlp_vectors.append(get_mlp_vector(X, rules))
        mlp_labels.append(1)
    for X in X_false:
        mlp_vectors.append(get_mlp_vector(X, rules))
        mlp_labels.append(0)
    
    return np.array(mlp_vectors), np.array(mlp_labels)


def validation(train_X_true, train_X_false, val_X_true, val_X_false):
    print("Now checking if we can go to next course")
    samples_per_rule = 3
    logic_output_dim = 50
    print("Now checking if we can go to next course")
    rules = get_rules_from_data(train_X_true, samples_per_rule, logic_output_dim)
    mlp_train_vectors, mlp_train_labels = get_mlp_data(train_X_true, train_X_false, rules)

    index = np.array(list(range(len(mlp_train_labels))))
    np.random.shuffle(index)
    mlp_train_vectors = mlp_train_vectors[index]
    mlp_train_labels = mlp_train_labels[index]
    
    best_accuracy = 0
    
    #Try three times to find the best mlp
    for _ in range(3):
        print("Training mlp...")
        
        ### TODO
        # mlp_model = get_mlp_net(logic_output_dim)
        # mlp_model.compile(loss='binary_crossentropy',
        #                   optimizer='rmsprop',
        #                   metrics=['accuracy'])
        # mlp_model.fit(mlp_train_vectors,
        #               mlp_train_labels,
        #               epochs=MLP_EPOCHS,
        #               batch_size=MLP_BATCHSIZE,
        #               verbose=0)
        #Prepare MLP validation data
        
        mlp_val_vectors, mlp_val_labels = get_mlp_data(val_X_true, val_X_false, rules)
        
        ## TODO
        #Get MLP validation result
        # result = mlp_model.evaluate(mlp_val_vectors,
        #                             mlp_val_labels,
        #                             batch_size=MLP_BATCHSIZE,
        #                             verbose=0)
        print("MLP validation result:", result)
        accuracy = result[1]

        if accuracy > best_accuracy:
            best_accuracy = accuracy
    return best_accuracy



def train_HED(model, abducer, train_data, test_data, epochs=50, select_num=10, verbose=-1):
    train_X, train_Z, train_Y = train_data
    test_X, test_Z, test_Y = test_data

    min_len = 5
    max_len = 8

    cp_threshold = 0.9
    blmp_threshold = 0.9

    cnt_threshold = 5
    acc_threshold = 0.86

    # Start training / for each length of equations
    for equation_len in range(min_len, max_len):

        ### TODO: get_data, e.g.
        # train_X_true = train_X['True'][equation_len]
        # train_X_true.append(train_X['True'][equation_len + 1])
        

        while True:
            # Abduce and train NN
            consistent_percentage, batch_label_model_precision = abduce_and_train(model, abducer, train_X_true, select_num)
            if consistent_percentage == 0:
                continue

            # Test if we can use mlp to evaluate
            if consistent_percentage >= cp_threshold and batch_label_model_precision >= blmp_threshold:
                condition_cnt += 1
            else:
                condition_cnt = 0
            # The condition has been satisfied continuously five times
            if condition_cnt >= cnt_threshold:
                best_accuracy = validation(train_X_true, train_X_false, val_X_true, val_X_false)

                # decide next course or restart
                if best_accuracy > acc_threshold:
                    # Save model and go to next course
                    ## TODO: model.save_weights()
                    break

                else:
                    # Restart current course: reload model
                    if equation_len == min_len:
                        ## TODO: model.set_weights(pretrain_model.get_weights())
                        model.set_weights()
                    else:
                        ## TODO: model.load_weights()
                        model.load_weights()
                    print("Failed! Reload model.")
                    condition_cnt = 0



    return model


if __name__ == "__main__":
    pass
