# -*- coding: utf-8 -*-


from const import (ANGLE_MAX, ANGLE_MIN, COLOR_MAX, COLOR_MIN, NUM_MAX,
                   NUM_MIN, SIZE_MAX, SIZE_MIN, TYPE_MAX, TYPE_MIN, UNI_MAX,
                   UNI_MIN)


def gen_layout_constraint(
        pos_type, pos_list,
        num_min=NUM_MIN, num_max=NUM_MAX,
        uni_min=UNI_MIN, uni_max=UNI_MAX):
    """
    Generate constraints for the layout.
    By default, the layout will have:
        - num: [0, 8] (uses 0-based indexing, so the true number will be num+1)
        - uni: [False, False, False, True] ?? (I guess these are the masks for each attribute)
    :param pos_type: ??
    :param pos_list: List of available positions to put the shapes in. Each element contains coordinates in the
                     following form:
                        - all but line: [x, y, width, height]
                        - line: [x_from, y_from, x_to, y_to]
    :param num_min: minimal number of shapes in the layout
    :param num_max: maximal number of shapes in the layout
    :param uni_min: ??
    :param uni_max: ??
    :return:
    """
    constraint = {
        "Number": [num_min, num_max],
        "Position": [pos_type, pos_list[:]],
        "Uni": [uni_min, uni_max]}
    return constraint


def gen_entity_constraint(
        type_min=TYPE_MIN, type_max=TYPE_MAX,
        size_min=SIZE_MIN, size_max=SIZE_MAX,
        color_min=COLOR_MIN, color_max=COLOR_MAX,
        angle_min=ANGLE_MIN, angle_max=ANGLE_MAX):
    """
    Generate constraints for each entity applicable to the given layout.
    By default, the entity will be of:
        - type: ["none", "triangle", "square", "pentagon", "hexagon", "circle"]
        - size: [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        - color: [255, 224, 196, 168, 140, 112, 84, 56, 28, 0]
        - angle: [-135, -90, -45, 0, 45, 90, 135, 180]
    :param type_min: minimal index of entity's type
    :param type_max: maximal index of entity's type
    :param size_min: minimal index of entity's size
    :param size_max: maximal index of entity's size
    :param color_min: minimal index of entity's color
    :param color_max: maximal index of entity's color
    :param angle_min: minimal index of entity's angle
    :param angle_max: maximal index of entity's angle
    :return:
    """
    constraint = {
        "Type": [type_min, type_max],
        "Size": [size_min, size_max],
        "Color": [color_min, color_max],
        "Angle": [angle_min, angle_max]}
    return constraint


def rule_constraint(
        rule_list, num_min, num_max,
        uni_min, uni_max,
        type_min, type_max,
        size_min, size_max,
        color_min, color_max):
    """Generate constraints given the rules and the original constraints
    from layout and entity. Note that each attribute has at most one rule
    applied on it.
    Arguments:
        rule_list(ordered list of Rule): all rules applied to this layout
        others (int): boundary levels for each attribute in a layout; note that
            num_max + 1 == len(layout.position.values)
    Returns:
        layout_constraint(dict): a new layout constraint
        entity_constraint(dict): a new entity constraint
    """
    assert len(rule_list) > 0
    for rule in rule_list:
        if rule.name == "Progression":
            # rule.value: add/sub how many levels
            if rule.attr == "Number":
                if rule.value > 0:
                    num_max = num_max - rule.value * 2
                else:
                    num_min = num_min - rule.value * 2
            if rule.attr == "Position":
                # Progression here means moving in Layout slots in order
                abs_value = abs(rule.value)
                num_max = num_max - abs_value * 2
            if rule.attr == "Type":
                if rule.value > 0:
                    type_max = type_max - rule.value * 2
                else:
                    type_min = type_min - rule.value * 2
            if rule.attr == "Size":
                if rule.value > 0:
                    size_max = size_max - rule.value * 2
                else:
                    size_min = size_min - rule.value * 2
            if rule.attr == "Color":
                if rule.value > 0:
                    color_max = color_max - rule.value * 2
                else:
                    color_min = color_min - rule.value * 2
        if rule.name == "Arithmetic":
            # rule.value > 0 if add col_0 + col_1
            # rule.value < 0 if sub col_0 - col_1
            if rule.attr == "Number":
                if rule.value > 0:
                    num_max = num_max - num_min - 1
                else:
                    num_min = 2 * num_min + 1
            if rule.attr == "Position":
                # SET_UNION
                # at least two position configurations
                if rule.value > 0:
                    num_max = num_max - 1
                # num_min makes sure of overlap
                # at least two configurations
                # SET_DIFF
                else:
                    num_min = (num_max + 2) / 2 - 1
                    num_max = num_max - 1
            if rule.attr == "Size":
                if rule.value > 0:
                    size_max = size_max - size_min - 1
                else:
                    size_min = 2 * size_min + 1
            if rule.attr == "Color":
                # at least two different colors
                if color_max - color_min < 1:
                    color_max = color_min - 1
                else:
                    if rule.value > 0:
                        color_max = color_max - color_min
                    if rule.value < 0:
                        color_min = 2 * color_min
        if rule.name == "Distribute_Three":
            # if less than 3 values, invalidate it
            if rule.attr == "Number":
                if num_max - num_min + 1 < 3:
                    num_max = num_min - 1
            if rule.attr == "Position":
                # max number allowed in the layout should be >= 3
                if num_max + 1 < 3:
                    num_max = num_min - 1
                # num_max + 1 == len(layout.position.values)
                # C_{num_max + 1}^{num_value} >= 3
                # C_{num_max + 1} = num_max + 1 >= 3
                # hence only need to constrain num_max: num_max = num_max - 1
                # Check Yang Hui’s Triangle (Pascal's Triangle): https://www.varsitytutors.com/hotmath/hotmath_help/topics/yang-huis-triangle
                else:
                    num_max = num_max - 1
            if rule.attr == "Type":
                if type_max - type_min + 1 < 3:
                    type_max = type_min - 1
            if rule.attr == "Size":
                if size_max - size_min + 1 < 3:
                    size_max = size_min - 1
            if rule.attr == "Color":
                if color_max - color_min + 1 < 3:
                    color_max = color_min - 1

    return gen_layout_constraint(None, [],
                                 num_min, num_max,
                                 uni_min, uni_max), \
        gen_entity_constraint(type_min, type_max,
                              size_min, size_max,
                              color_min, color_max)
