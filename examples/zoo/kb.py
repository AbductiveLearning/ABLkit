from z3 import Solver, Int, If, Not, Implies, Sum, sat
import openml
from abl.reasoning import KBBase

class ZooKB(KBBase):
    def __init__(self):
        super().__init__(pseudo_label_list=list(range(7)), use_cache=False)

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
            return total_weight.as_long()
        else:
            # No solution found
            return 1e10
