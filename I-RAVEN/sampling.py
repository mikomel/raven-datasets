# -*- coding: utf-8 -*-
from typing import List


import numpy as np
from scipy.special import comb

from const import RULE_ATTR
from Rule import Rule_Wrapper


def sample_rules(
    num_components: int,
    contains_mesh_component: bool,
    configuration: str,
    ood_attribute_indices: List[int] = [],
    set_name: str = "train",
    train_set_rules: List[str] = [],
):
    """First sample # components; for each component, sample a rule on each attribute."""
    assert len(ood_attribute_indices) == len(train_set_rules)
    is_out_of_distribution_dataset = len(ood_attribute_indices) > 0

    all_rules = []
    for component_idx in range(num_components):
        all_rules_component = []
        for j in range(len(RULE_ATTR)):

            if is_out_of_distribution_dataset:

                if j in ood_attribute_indices:
                    train_set_rule = train_set_rules[ood_attribute_indices.index(j)]
                    if do_enforce_train_set_rule(
                        set_name, configuration, component_idx, j, train_set_rule
                    ):
                        # RULE_ATTR[1:] (Type, Size, Color) will always have a single entry matching the train set rule.
                        # However, RULE_ATTR[0] (Number / Position) may have up to two entries matching the train set rule.
                        idx = np.random.choice(
                            [
                                i
                                for i, rule_attr in enumerate(RULE_ATTR[j])
                                if rule_attr[0] == train_set_rule
                            ]
                        )
                    else:
                        # Enforce rule to be other than the train set rule. In attributeless datasets, the training and
                        # validation matrices will have the missing attribute governed by the train set rule in each row,
                        # while the testing matrices, whenever applicable, will have a rule other than the train set rule,
                        # which governs the attribute.
                        idx = np.random.choice(
                            [
                                i
                                for i, rule in enumerate(RULE_ATTR[j])
                                if rule[0] != train_set_rule
                            ]
                        )

                else:
                    # Select a random rule that will govern the attribute
                    idx = np.random.choice(len(RULE_ATTR[j]))

            else:

                if (
                    contains_mesh_component
                    and j > 0
                    and component_idx == num_components - 1
                ):
                    # For the mesh component, Type, Size, and Color (j > 0) are always Constant (the last rule for each attribute)
                    idx = len(RULE_ATTR[j]) - 1

                else:
                    # Select a random rule that will govern the attribute
                    idx = np.random.choice(len(RULE_ATTR[j]))

            name_attr_param = RULE_ATTR[j][idx]
            all_rules_component.append(
                Rule_Wrapper(
                    name_attr_param[0],
                    name_attr_param[1],
                    name_attr_param[2],
                    component_idx=component_idx,
                )
            )
        all_rules.append(all_rules_component)
    return all_rules


def do_enforce_train_set_rule(
    set_name: str,
    configuration: str,
    component_idx: int,
    ood_attribute_idx: int,
    train_set_rule: str,
) -> bool:
    """
    In attributeless datasets the rule for an attribute will be enforced to the train set rule if:
    1. The attribute was selected to be ood (e.g. Color in A/Color)
    2. Dateset split is "train"
    3. The train set rule can be applied to the attribute of the component
      e.g. Progression may govern Color in the 0th component of the Center configuration,
      but it can't govern Color in the 1st component of the Out-InCenter configuration,
      as its outer shape has only a single value of the attribute available.
    See tests/test_sampling.py for detailed test-cases.
    component_idx == 0 -> Out
    component_idx == 1 -> In
    """
    is_position_ood = ood_attribute_idx == 0
    if is_position_ood:
        # Can modify Position in 2x2, 3x3 and in the Inner component of Out-InGrid
        can_modify_position = configuration in {"distribute_four", "distribute_nine"}
        can_modify_position |= (
            configuration == "in_distribute_four_out_center_single"
            and component_idx == 1
        )

        if set_name != "test" and train_set_rule == "Constant":
            return True

        if set_name != "test" and train_set_rule != "Constant":
            return can_modify_position

        if set_name == "test" and train_set_rule == "Constant":
            return not can_modify_position

        if set_name == "test" and train_set_rule != "Constant":
            return False

        raise ValueError(
            "Shouldn't get here."
            "do_enforce_train_set_rule"
            f"{set_name} {configuration} {component_idx} {ood_attribute_idx} {train_set_rule}"
        )

    is_size_ood = ood_attribute_idx == 2
    if is_size_ood:
        # Arithmetic on Size can't be applied to the Outer component in Out-In configurations
        if (
            set_name != "test"
            and train_set_rule == "Arithmetic"
            and configuration
            in {
                "in_center_single_out_center_single",
                "in_distribute_four_out_center_single",
            }
            and component_idx == 0
        ):
            return False

    is_color_ood = ood_attribute_idx == 3
    if is_color_ood:
        # Can't modify Color in the Outer component of In-Out configurations
        can_modify_color = not (
            configuration
            in {
                "in_center_single_out_center_single",
                "in_distribute_four_out_center_single",
            }
            and component_idx == 0
        )

        if set_name != "test" and train_set_rule == "Constant":
            return True

        if set_name != "test" and train_set_rule != "Constant":
            return can_modify_color

        if set_name == "test" and train_set_rule == "Constant":
            return not can_modify_color

        if set_name == "test" and train_set_rule != "Constant":
            return False

        raise ValueError(
            "Shouldn't get here."
            "do_enforce_train_set_rule"
            f"{set_name} {configuration} {component_idx} {ood_attribute_idx} {train_set_rule}"
        )

    return set_name != "test"


# pay attention to Position Arithmetic, new entities (resample)
def sample_available_attributes(rule_groups, row_3_3):
    """Sample available attributes whose values could be modified.
    Arguments:
        rule_groups(list of list of Rule): a list of rules to apply to the component
        row_3_3(AoTNode): the answer AoT
    Returns:
        ret(list of list): [component_idx, attr, available_times, constraints, attr_uni]
    """
    ret = []
    for i in range(len(rule_groups)):
        rule_group = rule_groups[i]
        start_node_layout = row_3_3.children[0].children[i].children[0]
        row_3_3_layout = row_3_3.children[0].children[i].children[0]
        uni = row_3_3_layout.uniformity.get_value()
        # Number/Position
        # If Rule on Number: Only change Number
        # If Rule on Position: Both Number and Position could be changed
        rule = rule_group[0]
        num = row_3_3_layout.number.get_value()
        most_num = len(start_node_layout.position.values)
        if rule.attr == "Number":
            num_times = 0
            min_level = start_node_layout.orig_layout_constraint["Number"][0]
            max_level = start_node_layout.orig_layout_constraint["Number"][1]
            for k in range(min_level, max_level + 1):
                if k + 1 != num:
                    num_times += comb(most_num, k + 1)
            if num_times > 0:
                ret.append([i, "Number", num_times, min_level, max_level, None])
        # Constant or on Position
        else:
            num_times = 0
            min_level = start_node_layout.orig_layout_constraint["Number"][0]
            max_level = start_node_layout.orig_layout_constraint["Number"][1]
            for k in range(min_level, max_level + 1):
                if k + 1 != num:
                    num_times += comb(most_num, k + 1)
            if num_times > 0:
                ret.append([i, "Number", num_times, min_level, max_level, None])
            pos_times = comb(most_num, row_3_3_layout.number.get_value())
            pos_times -= 1
            if pos_times > 0:
                ret.append([i, "Position", pos_times, None, None, None])
        # Type, Size, Color
        for j in range(1, len(rule_group)):
            rule = rule_group[j]
            rule_attr = rule.attr
            min_level = start_node_layout.orig_entity_constraint[rule_attr][0]
            max_level = start_node_layout.orig_entity_constraint[rule_attr][1]
            if rule.name == "Constant":
                if (
                    uni
                    or rule_group[0].name == "Constant"
                    or (
                        rule_group[0].attr == "Position"
                        and (
                            rule_group[0].name == "Progression"
                            or rule_group[0].name == "Distribute_Three"
                        )
                    )
                ):
                    times = max_level - min_level + 1
                    times = times - 1
                    if times > 0:
                        ret.append([i, rule_attr, times, min_level, max_level, uni])
            else:
                times = max_level - min_level + 1
                times = times - 1
                if times > 0:
                    ret.append([i, rule_attr, times, min_level, max_level, True])
    return ret


def sample_attribute(attributes):
    """Given the attr_avail list, sample one attribute to modify the value.
    If the available times becomes zero, delete it.
    Arguments:
        attributes(list of list): a flat component of available attributes
            to change the values; consisting of different component indexes
    """
    attribute_idx = np.random.choice(len(attributes))
    component_idx, attribute_name, _, min_level, max_level, _ = attributes[
        attribute_idx
    ]
    attributes[attribute_idx][2] -= 1
    if attributes[attribute_idx][2] == 0:
        del attributes[attribute_idx]
    return component_idx, attribute_name, min_level, max_level
