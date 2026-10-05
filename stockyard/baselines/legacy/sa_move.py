import copy
import random


class Plate:  # Modeling for each Plate
    def __init__(
        self,
        name,
        lotgrp,
        thickness,
        ship_type,
        planned_out,
        actual_out,
        input,
        pile_code=None,
    ):
        self.name = name
        self.lotgrp = lotgrp
        self.thickness = thickness
        self.ship_type = ship_type
        self.planned_out = planned_out
        self.actual_out = actual_out
        self.input = input

        self.pile_code = pile_code


class Pile:  # Modeling for Pile what will be stacked plates
    def __init__(self, name, capacity=float("inf"), type=None):
        self.name = name
        self.type = type
        self.capacity = capacity  # height

        self.piled_plate = list()
        self.piled_height = 0.0


def make_SA_sample(number_of_piles_from, number_of_piles_to):
    piles_from = dict()
    piles_to = dict()

    long = int(0.3 * number_of_piles_from)
    medium = int(0.45 * number_of_piles_from)
    short = number_of_piles_from - long - medium

    for i in range(long):
        piles_from["pile_from%i" % i] = Pile("pile_from%i" % i, 30, "Long")

    for i in range(long, long + medium):
        piles_from["pile_from%i" % i] = Pile("pile_from%i" % i, 30, "Medium")

    for i in range(long + medium, long + medium + short):
        piles_from["pile_from%i" % i] = Pile("pile_from%i" % i, 30, "Short")

    idx = 0

    all_plate_list = list()

    cap = 30
    max_out_date = 80

    n = int(cap * 3 / 4)

    for i in range(long):
        plate_list = list()
        # n = random.randint(18,23)
        for j in range(n):
            out_date = random.randint(0, max_out_date)
            plate = Plate(
                "plate%i" % idx,
                "x",
                1,
                "Long",
                out_date,
                out_date,
                0,
                "pile_from%i" % i,
            )
            plate_list.append(plate)
            all_plate_list.append(plate)
            idx += 1
        piles_from["pile_from%i" % i].piled_plate = plate_list

    for i in range(long, long + medium):
        plate_list = list()
        # n = random.randint(18,23)
        for j in range(n):
            out_date = random.randint(0, max_out_date)
            plate = Plate(
                "plate%i" % idx,
                "x",
                1,
                "Long",
                out_date,
                out_date,
                0,
                "pile_from%i" % i,
            )
            plate_list.append(plate)
            all_plate_list.append(plate)
            idx += 1
        piles_from["pile_from%i" % i].piled_plate = plate_list

    for i in range(long + medium, long + medium + short):
        plate_list = list()
        # n = random.randint(18,23)
        for j in range(n):
            out_date = random.randint(0, max_out_date)
            plate = Plate(
                "plate%i" % idx,
                "x",
                1,
                "Long",
                out_date,
                out_date,
                0,
                "pile_from%i" % i,
            )
            plate_list.append(plate)
            all_plate_list.append(plate)
            idx += 1
        piles_from["pile_from%i" % i].piled_plate = plate_list

    long = int(0.3 * number_of_piles_to)
    medium = int(0.45 * number_of_piles_to)
    short = number_of_piles_to - long - medium

    for i in range(long):
        piles_to["pile_to%i" % i] = Pile("pile_to%i" % i, 30, "Long")

    for i in range(long, long + medium):
        piles_to["pile_to%i" % i] = Pile("pile_to%i" % i, 30, "Medium")

    for i in range(long + medium, long + medium + short):
        piles_to["pile_to%i" % i] = Pile("pile_to%i" % i, 30, "Short")

    piles_to_list = [x for x in piles_to.keys()]
    piles_from_list = []
    for pile in piles_from.keys():
        for i in range(len(piles_from[pile].piled_plate)):
            piles_from_list.append(pile)

    from_to = [[x, random.choice(piles_to_list)] for x in piles_from_list]
    return from_to, piles_from, piles_to


def SA_move(data, piles_from, piles_to):
    max_num = 120

    reversal = 0

    piles_from_copy = copy.deepcopy(piles_from)
    piles_to_copy = copy.deepcopy(piles_to)

    for target_pile in data:
        if len(piles_to_copy[target_pile[1]].piled_plate) == max_num:
            reversal = 9999999999
            return reversal, piles_to_copy
        target_plate = piles_from_copy[target_pile[0]].piled_plate.pop()
        if (
            len(piles_to_copy[target_pile[1]].piled_plate) != 0
            and piles_to_copy[target_pile[1]].piled_plate[-1].actual_out
            < target_plate.actual_out
        ):
            reversal += 1
        piles_to_copy[target_pile[1]].piled_plate.append(target_plate)

    return reversal, piles_to_copy
