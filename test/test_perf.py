import collections
import csv
import time

import pytest
import torch
import numpy as np
import terminaltables
from tqdm import tqdm
import termcolor

from vectors import vector


def add(a, b):
    return a + b


def iadd(a, b):
    a += b
    return a


def sub(a, b):
    return a - b


def isub(a, b):
    a -= b
    return a


def mul(a, b):
    return a * b


def imul(a, b):
    a *= b
    return a


def div(a, b):
    return a / b


def idiv(a, b):
    a /= b
    return a


class Native:
    def __init__(self, a):
        self.a = a

    def __add__(self, other):
        return Native(self.a.copy()).__iadd__(other)

    def __iadd__(self, other):
        for i in range(len(self.a)):
            self.a[i] += other.a[i]
        return self

    def __mul__(self, other):
        return Native([a * b for a, b in zip(self.a, other.a)])

    def __imul__(self, other):
        for i in range(len(self.a)):
            self.a[i] *= other.a[i]
        return self

    def __sub__(self, other):
        return Native([a - b for a, b in zip(self.a, other.a)])

    def __isub__(self, other):
        for i in range(len(self.a)):
            self.a[i] -= other.a[i]
        return self

    def __truediv__(self, other):
        return Native([a / b for a, b in zip(self.a, other.a)])

    def __itruediv__(self, other):
        for i in range(len(self.a)):
            self.a[i] /= other.a[i]
        return self


operations = [add, iadd, sub, isub, mul, imul, div, idiv]

testers = [{
    "_id": "torch",
    "name": "Torch",
    "constructor": torch.tensor,
}, {
    "_id": "numpy",
    "name": "Numpy",
    "constructor": np.array,
}, {
    "_id": "EMVector",
    "name": "Extreme Math Vector",
    "constructor": vector,
}, {
    "_id": "native",
    "name": "Native Python",
    "constructor": Native,
    "max": 3,
}]


def score(n, m=1, k=1000):
    all_results = {}
    for tester in testers:
        if "max" in tester and tester["max"] < n:
            continue

        print(f"Starting {tester['name']}")
        results = {
            "generation": 0,
            "operations": collections.defaultdict(float),
        }

        a = [0.1] * (10**n)
        b = [0.2] * (10**n)
        for _ in tqdm(range(m)):
            for op in operations:
                # timed array generation
                start = time.time()
                ap = tester["constructor"](a)
                bp = tester["constructor"](b)
                results["generation"] += time.time() - start

                # timed computation
                start = time.time()
                for _ in range(k):
                    ap = op(ap, bp)
                results["operations"][op.__name__] += time.time() - start
        all_results[tester["_id"]] = results

    return all_results


def color(row):
    best = 0
    worst = 0
    for i, r in enumerate(row):
        if r < row[best]:
            best = i
        if r > row[worst]:
            worst = i

    row[worst] = termcolor.colored(str(row[worst]), 'red')
    row[best] = termcolor.colored(str(row[best]), 'green')

    return row


@pytest.mark.last
def test_perf():
    for i in range(1, 7):
        n = i
        m = 5

        results = score(n, m)

        ids = [tester["_id"] for tester in testers]

        table_data = []
        table_data.append(["operation"] + ids)
        table_data.append(["_gen_"] +
                          [results[id]["generation"] for id in results.keys()])
        for op in operations:
            res = [
                results[id]["operations"][op.__name__]
                for id in results.keys()
            ]
            table_data.append([op.__name__] + res)

        with open(f".results/{n}.csv", "w+") as my_csv:
            csvWriter = csv.writer(my_csv, delimiter=',')
            csvWriter.writerows(table_data)

        table_data = [table_data[0]] + [[row[0]] + color(row[1:])
                                        for row in table_data[1:]]

        table = terminaltables.AsciiTable(table_data)
        print(table.table)

    return
