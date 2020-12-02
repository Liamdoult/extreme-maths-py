import collections
import csv
import time

import torch
import numpy as np
import terminaltables
from tqdm import tqdm
import termcolor

from extreme_maths.vectors import EMVector
from extreme_maths.vectors import EMVectorCuda
from extreme_maths.vectors import EMVectorOCL
from extreme_maths.vectors import EMVectorThreaded


def add(a, b):
    c = a + b


def iadd(a, b):
    a += b


def sub(a, b):
    c = a + b


def isub(a, b):
    a += b


def mul(a, b):
    c = a + b


def imul(a, b):
    a += b


def div(a, b):
    c = a + b


def idiv(a, b):
    a += b


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
    "constructor": EMVector,
}, {
    "_id": "EMVectorCuda",
    "name": "Extreme Math Vector CUDA",
    "constructor": EMVectorCuda,
}, {
    "_id": "EMVectorOCL",
    "name": "Extreme Math Vector OpenCL",
    "constructor": EMVectorOCL,
}, {
    "_id": "EMVectorThreaded",
    "name": "Extreme Math Vector Threaded",
    "constructor": EMVectorThreaded,
}]


def score(n):
    all_results = {}
    for tester in testers:
        print(f"Starting {tester['name']}")
        results = {
            "generation": 0,
            "operations": collections.defaultdict(float),
        }

        for i in tqdm(range(n)):
            for _ in range(n - i):
                a = [1.0] * (10**i)
                b = [2.0] * (10**i)

                for op in operations:
                    # timed array generation
                    start = time.time()
                    ap = tester["constructor"](a)
                    bp = tester["constructor"](b)
                    results["generation"] += time.time() - start

                    # timed computation
                    start = time.time()
                    res = op(ap, bp)
                    results["operations"][op.__name__] += time.time() - start
        all_results[tester["_id"]] = results

    return all_results


def color(row):
    best = 0
    worst = 0
    for i, r in enumerate(row[1:]):
        if r < row[best]:
            best = i
        if r > row[worst]:
            worst = i

    row[worst] = termcolor.colored(str(row[worst]), 'red')
    row[best] = termcolor.colored(str(row[best]), 'green')

    return row


def test_perf():
    n = 5

    results = score(n)

    ids = [tester["_id"] for tester in testers]

    table_data = []
    table_data.append(["operation"] + ids)
    table_data.append(["_gen_"] + [results[id]["generation"] for id in ids])
    for op in operations:
        res = [results[id]["operations"][op.__name__] for id in ids]
        table_data.append([op.__name__] + res)

    with open(".results.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(table_data)

    table_data = [table_data[0]] + [[row[0]] + color(row[1:])
                                    for row in table_data[1:]]

    table = terminaltables.AsciiTable(table_data)
    print(table.table)

    return
