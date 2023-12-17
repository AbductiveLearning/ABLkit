pseudo_label(N) :- between(0, 9, N).
logic_forward([Z1, Z2], Res) :- pseudo_label(Z1), pseudo_label(Z2), Res is Z1+Z2.
