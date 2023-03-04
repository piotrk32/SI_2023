import math

import pandas as pd
import numpy as np
import csv

# 3a
last_col = 'Exited'
with open('Churn_Modelling.csv', 'r') as file:
    read = csv.DictReader(file)
    class_name = set(row[last_col] for row in read)
    print(f"Liczba klas w kolumnie: {len(class_name)}")
    print(f"Nazwy klas: {', '.join(class_name)}")

# 3b
count0 = 0
count1 = 0

with open('Churn_Modelling.csv', 'r') as file:
    read = csv.DictReader(file)
    for row in read:
        if row[last_col] == '0':
            count0 += 1
        elif row[last_col] == '1':
            count1 += 1

print(f"Ilosc wystapien 0: {count0}")
print(f"Ilosc wystapien 1: {count1}")

# 3c

customer_id_col = 'CustomerId'
with open('Churn_Modelling.csv', 'r') as file:
    read = csv.DictReader(file)

    min_value = float('inf')
    max_value = float('-inf')

    for row in read:
        value = int(row[customer_id_col])
        if value < min_value:
            min_value = value
        if value > max_value:
            max_value = value

    print(f"Najmniejsza wartość: {min_value}")
    print(f"Największa wartość: {max_value}")

# zad3d
df = pd.read_csv('Churn_Modelling.csv')
print('Liczba różnych wartości dla każdego atrybutu:')
print(df.nunique())

# zad3e
df = pd.read_csv('Churn_Modelling.csv')
print('Lista wszystkich różnych wartości dla każdego atrybutu:')
for col in df.columns:
    print(col)
    print(df[col].unique())

# zad3f
df = pd.read_csv('Churn_Modelling.csv')
print('Odchylenie standardowe dla poszczególnych atrybutów:')
for col in df.columns:
    if df[col].dtype == 'float64':
        print(f'Cały plik CSV: {df[col].std()}')
        print(f'Klasy decyzyjne: {df.groupby("Exited")[col].std()}\n')

# zad4a
df = pd.read_csv('Churn_Modelling.csv')

col_count = df.shape[0]
miss_col = int(col_count * 0.1)
random_miss_col = np.random.choice(col_count, size=miss_col, replace=False)

for col in df.columns:
    df.loc[random_miss_col, col] = np.nan

    if df[col].dtype == 'float64':
        atrybut_numeryczny = df[col].mean()
        df[col].fillna(atrybut_numeryczny, inplace=True)
    else:
        atrybut_symboliczny = df[col].mode()[0]
        df[col].fillna(atrybut_symboliczny, inplace=True)
print(df.head(10))


# zad4b
def normalize_attribute_value(path, index, interval_start, interval_end):
    data = []
    with open(path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        data.append(header)
        min_val = float('inf')
        max_val = float('-inf')
        for row in reader:
            attribute_value = float(row[index])
            if attribute_value < min_val:
                min_val = attribute_value
            if attribute_value > max_val:
                max_val = attribute_value
            data.append(row)

    normalized_data = []
    normalized_data.append(header)
    for row in data[1:]:
        attribute_value = float(row[index])
        normalized_value = ((attribute_value - min_val) * (interval_end - interval_start) / (
            max_val - min_val)) + interval_start
        row[index] = normalized_value
        normalized_data.append(row)

    with open('znormalizowane_dane.csv', 'w', newline='') as file:
        save = csv.writer(file)
        save.writerows(normalized_data)

    print(
        f"Atrybut  {header[index]} znormalizowany do przedziału ({interval_start}, {interval_end})")


normalize_attribute_value('Churn_Modelling.csv', 1, -1, 1)  # interval <-1,1>


# zad4c
def standarize(path, index):
    with open(path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)
        data = list(reader)

        num_index = []
        for i, col in enumerate(header):
            if i != index and all(isinstance(row[i], (int, float)) for row in data):
                num_index.append(i)

        avg = [sum(float(row[i]) for row in data) / len(data) for i in num_index]
        standard_dev = [math.sqrt(sum((float(row[i]) - avg[j]) ** 2 for row in data) / len(data)) for j, i in
                    enumerate(num_index)]

        for i, row in enumerate(data):
            for j, index in enumerate(num_index):
                row[index] = (float(row[index]) - avg[j]) / standard_dev[j]
            data[i] = row

    with open('zad4c.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(data)

    print(f"Atrybut {header[index]} zestandaryzowany")


standarize('Churn_Modelling.csv', 4)

# zad4d
data = pd.read_csv('dane\Churn_Modelling.csv')

value_exists = pd.get_dummies(data['Geography'], prefix='Geography')
data = pd.concat([data, value_exists], axis=1)

data = data.drop(columns=['Geography_Germany'])

print(data.head())
