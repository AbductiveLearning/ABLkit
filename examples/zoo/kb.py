import openml
from z3 import If, Implies, Int, Not, Solver, Sum, sat

from ablkit.reasoning import KBBase


class ZooKB(KBBase):
    def __init__(self):
        super().__init__(pseudo_label_list=list(range(7)), use_cache=False)

        self.solver = Solver()

        dataset = openml.datasets.get_dataset(
            dataset_id=62,
            download_data=False,
            download_qualities=False,
            download_features_meta_data=False,
        )
        _, y, _, attribute_names = dataset.get_data(
            target=dataset.default_target_attribute
        )
        self.attribute_names = attribute_names
        self.target_names = y.cat.categories.tolist()

        # Create a Z3 Int variable for every attribute and target, keyed by name.
        self.vars = {name: Int(name) for name in self.attribute_names + self.target_names}
        v = self.vars

        rules = [
            Implies(v["milk"] == 1, v["mammal"] == 1),
            Implies(v["mammal"] == 1, v["milk"] == 1),
            Implies(v["mammal"] == 1, v["backbone"] == 1),
            Implies(v["mammal"] == 1, v["breathes"] == 1),
            Implies(v["feathers"] == 1, v["bird"] == 1),
            Implies(v["bird"] == 1, v["feathers"] == 1),
            Implies(v["bird"] == 1, v["eggs"] == 1),
            Implies(v["bird"] == 1, v["backbone"] == 1),
            Implies(v["bird"] == 1, v["breathes"] == 1),
            Implies(v["bird"] == 1, v["legs"] == 2),
            Implies(v["bird"] == 1, v["tail"] == 1),
            Implies(v["reptile"] == 1, v["backbone"] == 1),
            Implies(v["reptile"] == 1, v["breathes"] == 1),
            Implies(v["reptile"] == 1, v["tail"] == 1),
            Implies(v["fish"] == 1, v["aquatic"] == 1),
            Implies(v["fish"] == 1, v["toothed"] == 1),
            Implies(v["fish"] == 1, v["backbone"] == 1),
            Implies(v["fish"] == 1, Not(v["breathes"] == 1)),
            Implies(v["fish"] == 1, v["fins"] == 1),
            Implies(v["fish"] == 1, v["legs"] == 0),
            Implies(v["fish"] == 1, v["tail"] == 1),
            Implies(v["amphibian"] == 1, v["eggs"] == 1),
            Implies(v["amphibian"] == 1, v["aquatic"] == 1),
            Implies(v["amphibian"] == 1, v["backbone"] == 1),
            Implies(v["amphibian"] == 1, v["breathes"] == 1),
            Implies(v["amphibian"] == 1, v["legs"] == 4),
            Implies(v["insect"] == 1, v["eggs"] == 1),
            Implies(v["insect"] == 1, Not(v["backbone"] == 1)),
            Implies(v["insect"] == 1, v["legs"] == 6),
            Implies(v["invertebrate"] == 1, Not(v["backbone"] == 1)),
        ]
        self.weights = {rule: 1 for rule in rules}
        self.total_violation_weight = Sum(
            [If(Not(rule), self.weights[rule], 0) for rule in self.weights]
        )

    def logic_forward(self, pseudo_label, data_point):
        pseudo_label, data_point = pseudo_label[0], data_point[0]
        self.solver.reset()

        for name, value in zip(self.attribute_names, data_point):
            self.solver.add(self.vars[name] == value)
        for cate, name in zip(self.pseudo_label_list, self.target_names):
            value = 1 if cate == pseudo_label else 0
            self.solver.add(self.vars[name] == value)

        if self.solver.check() == sat:
            model = self.solver.model()
            return model.evaluate(self.total_violation_weight).as_long()
        # No solution found
        return 1e10
