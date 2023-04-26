import itertools


#zad1
def check_consistency(rules, decision_system):
    for rule in rules:
        for row in decision_system:
            if all(rule[i] == row[i] for i in rule):
                if rule[-1] != row[-1]:
                    return False
    return True

def find_rules(decision_system):
    attributes = len(decision_system[0]) - 1
    rules = []

    while decision_system:
        found = False
        for row in decision_system:
            for length in range(1, attributes + 1):
                if found:
                    break
                for combination in itertools.combinations(range(attributes), length):
                    rule = {i: row[i] for i in combination}
                    rule[-1] = row[-1]
                    if check_consistency([rule], decision_system):
                        rules.append(rule)
                        decision_system = [r for r in decision_system if not all(rule[i] == r[i] for i in rule)]
                        found = True
                        break

    return rules

decision_system = [
    [1, 1, 1, 1, 3, 1, 1],
    [1, 1, 1, 1, 3, 2, 1],
    [1, 1, 1, 3, 2, 1, 0],
    [1, 1, 1, 3, 3, 2, 1],
    [1, 1, 2, 1, 2, 1, 0],
    [1, 1, 2, 1, 2, 2, 1],
    [1, 1, 2, 2, 3, 1, 0],
    [1, 1, 2, 2, 4, 1, 1]
]

rules = find_rules(decision_system)
print("Reguły:")
for rule in rules:
    print(rule)

#zad2

# Stworzenie koniunkcji atrybutów dla wszystkich kombinacji:
# a1, a2, a3, a4, a5, a6, a1 ∧ a2, a1 ∧ a3, ..., a5 ∧ a6, a1 ∧ a2 ∧ a3, ..., a4 ∧ a5 ∧ a6, ..., a1 ∧ a2 ∧ a3 ∧ a4 ∧ a5 ∧ a6
#
# Wygenerowanie reguł dla każdej z koniunkcji:
# (a1=1) ⇒ (d=1), (a2=1) ⇒ (d=1), ... , (a1=1) ∧ (a2=1) ⇒ (d=1), ...
#
# Usunięcie reguł sprzecznych i wybór reguł o największym pokryciu.
#
# W wyniku zastosowania tej metody otrzymujemy następujące reguły dla systemu decyzyjnego:
#
# (a6=2) ⇒ (d=1), które pokrywają obiekty: o2, o4, o6
# (a5=4) ⇒ (d=1), które pokrywają obiekty: o8
# (a1=1) ∧ (a2=1) ∧ (a3=1) ∧ (a4=1) ⇒ (d=1), które pokrywają obiekty: o1
# (a3=1) ∧ (a5=2) ⇒ (d=0), które pokrywają obiekty: o3
# (a5=2) ∧ (a6=1) ⇒ (d=0), które pokrywają obiekty: o5
# (a3=2) ∧ (a5=3) ⇒ (d=0), które pokrywają obiekty: o7
# Jak można zauważyć, znalezione reguły są takie same jak te opisane w przykładzie 1. Oznacza to, że zastosowana metoda sekwencyjnego pokrycia znajduje takie same reguły, jak metoda opisana w treści zadania.
