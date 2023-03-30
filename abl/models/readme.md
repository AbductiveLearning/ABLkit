# `basic_model.py`

可以使用`basic_model.py`中实现的`BasicModel`类将`pytorch`神经网络模型包装成`sklearn`模型的形式.

## BasicModel 类提供的接口

|  方法   | 功能  |
|  ----  | ----  |
| fit(X, y) | 训练神经网络 |
| predict(X) | 预测 X 的类别 |
| predict_proba(X) | 预测 X 的类别概率 |
| score(X, y) | 计算模型在测试数据上的准确率 |
| save() | 保存模型 |
| load() | 加载模型 |


## BasicModel 类的参数

**model : torch.nn.Module** 
+ The PyTorch model to be trained or used for prediction.

**batch_size : int**
+ The batch size used for training.

**num_epochs : int**
+ The number of epochs used for training.

**stop_loss : Optional[float]**
+ The loss value at which to stop training.

**num_workers : int**
+ The number of workers used for loading data.

**criterion : torch.nn.Module**
+ The loss function used for training.

**optimizer : torch.nn.Module**
+ The optimizer used for training.

**transform : Callable[..., Any]**
+ The transformation function used for data augmentation.

**device : torch.device**
+ The device on which the model will be trained or used for prediction.

**recorder : Any**
+ The recorder used to record training progress.

**save_interval : Optional[int]**
+ The interval at which to save the model during training.

**save_dir : Optional[str]**
+ The directory in which to save the model during training.

**collate_fn : Callable[[List[T]], Any]**
+ The function used to collate data.

## 例子
>
> ```python
> # Three necessary component
> cls = LeNet5()
> criterion = nn.CrossEntropyLoss()
> optimizer = torch.optim.Adam(cls.parameters())
> 
> # Initialize base_model
> base_model = BasicModel(
>     cls,
>     criterion,
>     optimizer,
>     torch.device("cuda:0"),
>     batch_size=32,
>     num_epochs=10,
> )
>
> # Prepare data
> train_X, train_y = get_train_data()
> test_X, test_y = get_test_data()
> 
> # Train model
> base_model.fit(train_X, train_y)
>
> # Predict
> base_model.predict(test_X)
>
> # Validation
> base_model.score(test_X, test_y)
> ```

# `wabl_models.py`

`wabl_models.py`中实现的`WABLBasicModel`能够序列化数据并为不同的机器学习模型提供统一的接口.

## WABLBasicModel 类提供的接口

|  方法   | 功能  |
|  ----  | ----  |
| train(X, Y) | 利用训练数据训练机器学习模型（不涉及反绎） |
| predict(X) | 预测 X 的类别和概率 |
| valid(X, Y) | 计算模型在测试数据上的准确率 |

## WABLBasicModel 类的参数
**base_model : Machine Learning Model**
+ The base model to use for training and prediction.

**pseudo_label_list : List[Any]**
+ A list of pseudo labels to use for training.

## 序列化数据
考虑到训练数据可能多种组织形式，比如：\
`X: List[List[img]], Y: List[List[label]]`\
`X: List[List[img]], Y: List[label]`\
`X: List[img], Y: List[label]`
... \
不便于训练. 因此先将形式统一为：`X: List[img], Y: List[label]`，也就是所谓的序列化数据.

## 例子
>
> ```python
> # Three necessary component
> # 'ml_model' is no longer limited to NN models
> model = WABLBasicModel(ml_model, kb.pseudo_label_list)
>
> # Prepare data
> train_X, train_y = get_train_data()
> test_X, test_y = get_test_data()
> 
> # Train model
> model.train(train_X, train_y)
>
> # Predict
> model.predict(test_X)
>
> # Validation
> model.valid(test_X, test_y)
> ```