"""Exact capacity allocation for eight synthetic procedures; no external solver."""
from itertools import product
import json

PROCEDURES = [('A', 3, 8), ('B', 3, 8), ('C', 2, 7), ('D', 2, 7),
              ('E', 2, 6), ('F', 1, 4), ('G', 1, 4), ('H', 1, 3)]


def objective(assignment, procedures=PROCEDURES, capacity=4):
    loads = [sum(p[1] for p,a in zip(procedures, assignment) if a == room) for room in (1,2)]
    if any(load > capacity for load in loads):
        return -1
    return sum(p[2] for p,a in zip(procedures, assignment) if a), loads


def solve(procedures=PROCEDURES, capacity=4):
    best = (-1, (), [])
    # Each procedure is either deferred (0), or assigned to room 1 or 2.
    for assignment in product((0,1,2), repeat=len(procedures)):
        result = objective(assignment, procedures, capacity)
        if result != -1 and result[0] > best[0]:
            best = (result[0], assignment, result[1])
    return best


def first_fit(procedures=PROCEDURES, capacity=4):
    loads, assignment = [0,0], []
    for _, duration, _ in procedures:
        room = next((i for i in range(2) if loads[i] + duration <= capacity), None)
        assignment.append(0 if room is None else room+1)
        if room is not None: loads[room] += duration
    return objective(assignment, procedures, capacity)[0]


if __name__ == '__main__':
    score, assignment, loads = solve()
    print(json.dumps({'data': 'synthetic; two rooms, four hours per room',
                      'first_fit_priority_score': first_fit(), 'optimal_priority_score': score,
                      'room_loads_hours': loads,
                      'assignment': dict(zip((p[0] for p in PROCEDURES), assignment))}, indent=2))
