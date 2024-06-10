# -*- coding: utf-8 -*-


import argparse
import copy
import os
import random
from typing import List

import numpy as np

from build_tree import (
    build_center_single,
    build_distribute_four,
    build_distribute_nine,
    build_in_center_single_out_center_single,
    build_in_distribute_four_out_center_single,
    build_left_center_single_right_center_single,
    build_up_center_single_down_center_single,
)
from matplotlib import pyplot as plt
from const import IMAGE_SIZE
from rendering import render_panel
from sampling import sample_available_attributes, sample_rules
from serialize import (
    dom_problem,
    serialize_aot,
    serialize_rules,
    serialize_modifications,
)
from solver import solve


def merge_component(dst_aot, src_aot, component_idx):
    src_component = src_aot.children[0].children[component_idx]
    dst_aot.children[0].children[component_idx] = src_component


def separate(args, all_configs):
    random.seed(args.seed)
    np.random.seed(args.seed)

    should_render_random_mesh_component = args.mesh == 1
    contains_mesh_component = args.mesh == 2

    ood_attribute_idx = -1
    for i, is_iid_attribute in enumerate(
        [args.position, args.type, args.size, args.color]
    ):
        if not is_iid_attribute:
            ood_attribute_idx = i
            break

    for configuration in all_configs.keys():
        acc = 0
        for k in range(args.num_samples):
            count_num = k % 10
            if count_num < (10 - args.val - args.test):
                set_name = "train"
            elif count_num < (10 - args.test):
                set_name = "val"
            else:
                set_name = "test"

            root = all_configs[configuration]
            # num_components can be used to determine for which components rules should be sampled
            num_components = len(root.children[0].children)
            while True:
                rule_groups = sample_rules(
                    num_components,
                    contains_mesh_component,
                    configuration,
                    ood_attribute_idx,
                    set_name,
                )
                new_root = root.prune(rule_groups)
                if new_root is not None:
                    break

            start_node = new_root.sample()

            row_1_1 = copy.deepcopy(start_node)
            for l in range(len(rule_groups)):
                rule_group = rule_groups[l]
                rule_num_pos = rule_group[0]
                row_1_2 = rule_num_pos.apply_rule(row_1_1)
                row_1_3 = rule_num_pos.apply_rule(row_1_2)
                for i in range(1, len(rule_group)):
                    rule = rule_group[i]
                    row_1_2 = rule.apply_rule(row_1_1, row_1_2)
                for i in range(1, len(rule_group)):
                    rule = rule_group[i]
                    row_1_3 = rule.apply_rule(row_1_2, row_1_3)
                if l == 0:
                    to_merge = [row_1_1, row_1_2, row_1_3]
                else:
                    merge_component(to_merge[1], row_1_2, l)
                    merge_component(to_merge[2], row_1_3, l)
            row_1_1, row_1_2, row_1_3 = to_merge

            row_2_1 = copy.deepcopy(start_node)
            row_2_1.resample(True)
            for l in range(len(rule_groups)):
                rule_group = rule_groups[l]
                rule_num_pos = rule_group[0]
                row_2_2 = rule_num_pos.apply_rule(row_2_1)
                row_2_3 = rule_num_pos.apply_rule(row_2_2)
                for i in range(1, len(rule_group)):
                    rule = rule_group[i]
                    row_2_2 = rule.apply_rule(row_2_1, row_2_2)
                for i in range(1, len(rule_group)):
                    rule = rule_group[i]
                    row_2_3 = rule.apply_rule(row_2_2, row_2_3)
                if l == 0:
                    to_merge = [row_2_1, row_2_2, row_2_3]
                else:
                    merge_component(to_merge[1], row_2_2, l)
                    merge_component(to_merge[2], row_2_3, l)
            row_2_1, row_2_2, row_2_3 = to_merge

            row_3_1 = copy.deepcopy(start_node)
            row_3_1.resample(True)
            for l in range(len(rule_groups)):
                rule_group = rule_groups[l]
                rule_num_pos = rule_group[0]
                row_3_2 = rule_num_pos.apply_rule(row_3_1)
                row_3_3 = rule_num_pos.apply_rule(row_3_2)
                for i in range(1, len(rule_group)):
                    rule = rule_group[i]
                    row_3_2 = rule.apply_rule(row_3_1, row_3_2)
                for i in range(1, len(rule_group)):
                    rule = rule_group[i]
                    row_3_3 = rule.apply_rule(row_3_2, row_3_3)
                if l == 0:
                    to_merge = [row_3_1, row_3_2, row_3_3]
                else:
                    merge_component(to_merge[1], row_3_2, l)
                    merge_component(to_merge[2], row_3_3, l)
            row_3_1, row_3_2, row_3_3 = to_merge

            imgs = [
                render_panel(row_1_1, should_render_random_mesh_component),
                render_panel(row_1_2, should_render_random_mesh_component),
                render_panel(row_1_3, should_render_random_mesh_component),
                render_panel(row_2_1, should_render_random_mesh_component),
                render_panel(row_2_2, should_render_random_mesh_component),
                render_panel(row_2_3, should_render_random_mesh_component),
                render_panel(row_3_1, should_render_random_mesh_component),
                render_panel(row_3_2, should_render_random_mesh_component),
                np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8),
            ]
            context = [
                row_1_1,
                row_1_2,
                row_1_3,
                row_2_1,
                row_2_2,
                row_2_3,
                row_3_1,
                row_3_2,
            ]
            modifiable_attributes = sample_available_attributes(rule_groups, row_3_3)
            answer_AoT = copy.deepcopy(row_3_3)
            candidates = [answer_AoT]

            num_attributes_to_modify = 3
            selected_attr = select_modifiable_attributes(
                num_attributes_to_modify,
                contains_mesh_component,
                modifiable_attributes,
                num_components,
            )
            random.shuffle(selected_attr)

            mode = None
            # switch attribute 'Number' for convenience
            pos = [
                i for i in range(len(selected_attr)) if selected_attr[i][1] == "Number"
            ]
            if pos:
                pos = pos[0]
                selected_attr[pos], selected_attr[-1] = (
                    selected_attr[-1],
                    selected_attr[pos],
                )

                pos = [
                    i
                    for i in range(len(selected_attr))
                    if selected_attr[i][1] == "Position"
                ]
                if pos:
                    mode = "Position-Number"
            values = []
            if len(selected_attr) >= 3:
                mode_3 = None
                if mode == "Position-Number":
                    mode_3 = "3-Position-Number"
                for i in range(num_attributes_to_modify):
                    component_idx, attr_name, _, min_level, max_level, attr_uni = (
                        selected_attr[i]
                    )
                    value = answer_AoT.sample_new_value(
                        component_idx, attr_name, min_level, max_level, attr_uni, mode_3
                    )
                    values.append(value)
                    tmp = []
                    for j in candidates:
                        new_AoT = copy.deepcopy(j)
                        new_AoT.apply_new_value(component_idx, attr_name, value)
                        tmp.append(new_AoT)
                    candidates += tmp

            elif len(selected_attr) == 2:
                component_idx, attr_name, min_level, max_level, attr_uni = (
                    selected_attr[0][0],
                    selected_attr[0][1],
                    selected_attr[0][3],
                    selected_attr[0][4],
                    selected_attr[0][5],
                )
                value = answer_AoT.sample_new_value(
                    component_idx, attr_name, min_level, max_level, attr_uni, None
                )
                values.append(value)
                new_AoT = copy.deepcopy(answer_AoT)
                new_AoT.apply_new_value(component_idx, attr_name, value)
                candidates.append(new_AoT)
                component_idx, attr_name, min_level, max_level, attr_uni = (
                    selected_attr[1][0],
                    selected_attr[1][1],
                    selected_attr[1][3],
                    selected_attr[1][4],
                    selected_attr[1][5],
                )
                if mode == "Position-Number":
                    ran, qu = 6, 1
                else:
                    ran, qu = 3, 2
                for i in range(ran):
                    value = answer_AoT.sample_new_value(
                        component_idx, attr_name, min_level, max_level, attr_uni, None
                    )
                    values.append(value)
                    for j in range(qu):
                        new_AoT = copy.deepcopy(candidates[j])
                        new_AoT.apply_new_value(component_idx, attr_name, value)
                        candidates.append(new_AoT)

            elif len(selected_attr) == 1:
                component_idx, attr_name, min_level, max_level, attr_uni = (
                    selected_attr[0][0],
                    selected_attr[0][1],
                    selected_attr[0][3],
                    selected_attr[0][4],
                    selected_attr[0][5],
                )
                for i in range(7):
                    value = answer_AoT.sample_new_value(
                        component_idx, attr_name, min_level, max_level, attr_uni, None
                    )
                    values.append(value)
                    new_AoT = copy.deepcopy(answer_AoT)
                    new_AoT.apply_new_value(component_idx, attr_name, value)
                    candidates.append(new_AoT)

            random.shuffle(candidates)
            answers = []
            mods = []
            for candidate in candidates:
                answers.append(
                    render_panel(candidate, should_render_random_mesh_component)
                )
                mods.append(candidate.modified_attr)

            # imsave(generate_matrix_answer(imgs + answers), "/media/dsg3/hs/RAVEN_image/experiments2/{}/{}.jpg".format(key, k))

            image = imgs[0:8] + answers
            target = candidates.index(answer_AoT)
            predicted = solve(rule_groups, context, candidates)
            is_mesh_present = start_node.children[0].children[-1].name == "Mesh"
            max_components = len(start_node.children[0].children)
            meta_matrix, meta_target = serialize_rules(rule_groups, is_mesh_present)
            structure, meta_structure = serialize_aot(start_node)
            modifications_matrix = serialize_modifications(
                mods, is_mesh_present, max_components
            )
            np.savez(
                "{}/{}/RAVEN_{}_{}.npz".format(
                    args.save_dir, configuration, k, set_name
                ),
                image=image,
                target=target,
                predict=target,
                meta_matrix=meta_matrix,
                meta_target=meta_target,
                structure=structure,
                meta_structure=meta_structure,
                meta_answer_mods=modifications_matrix,
            )

            with open(
                "{}/{}/RAVEN_{}_{}.xml".format(
                    args.save_dir, configuration, k, set_name
                ),
                "wb",
            ) as f:
                dom = dom_problem(context + candidates, rule_groups)
                f.write(dom)

            # show_rpm(image)
            # print_rule(meta_matrix)

            if target == predicted:
                acc += 1
        # TODO: heuristics search is not implemented for the Mesh component
        print(f"Accuracy of {configuration}: {float(acc) / args.num_samples}")


RULES = ["Constant", "Progression", "Arithmetic", "Distribute_Three"]
ATTRIBUTES = ["Number", "Position", "Type", "Size", "Color"]


def print_rule(meta_matrix, num_components: int = 3, num_rules: int = 4) -> None:
    decoding = {}
    for component in range(num_components):
        decoding[component] = {}
        for row in meta_matrix[component * 4 : (component + 1) * 4]:
            rule = row[:num_rules].argmax()
            attributes = np.where(row[num_rules:] == 1)[0]
            for attribute in attributes:
                decoding[component][ATTRIBUTES[attribute]] = RULES[rule]
    print(decoding)


def show_rpm(image):
    m = 10
    canvas = np.ones((5 * 160 + 4 * m + 3, 4 * 160 + 3 * m + 3)) * 255
    for i, im in enumerate(image[:8]):
        row = i // 3
        col = i % 3
        x = 80 + col * (160 + m)
        y = 2 + row * (160 + m)
        canvas[y : y + 160, x : x + 160] = im
        add_border(canvas, x, y)
    for i, im in enumerate(image[8:]):
        row = i // 4
        col = i % 4
        x = col * (160 + m)
        y = 2 + 3 * (160 + m) + row * (160 + m)
        canvas[y : y + 160, x : x + 160] = im
        add_border(canvas, x, y)
    plt.imshow(canvas, cmap="gray")
    plt.axis("off")
    plt.show()


def add_border(canvas, x, y):
    canvas[y : y + 160, x : x + 1] = 0
    canvas[y : y + 160, x + 160 : x + 160 + 1] = 0
    canvas[y : y + 1, x : x + 160] = 0
    canvas[y + 160 : y + 160 + 1, x : x + 160] = 0


def select_modifiable_attributes(
    num_attributes_to_modify: int,
    contains_mesh_component: bool,
    modifiable_attributes: List,
    num_components: int,
) -> List:
    if num_attributes_to_modify < len(modifiable_attributes):
        if contains_mesh_component:
            mesh_component_idx = num_components - 1
            selected_attributes = []

            # Select modifiable Mesh attributes
            modifiable_mesh_attributes = [
                modifiable_attribute
                for modifiable_attribute in modifiable_attributes
                if modifiable_attribute[0] == mesh_component_idx
            ]
            num_selected_mesh_attributes = np.random.randint(
                1, len(modifiable_mesh_attributes) + 1
            )
            selected_mesh_attribute_indices = np.random.choice(
                len(modifiable_mesh_attributes),
                num_selected_mesh_attributes,
                replace=False,
            )
            selected_attributes += [
                modifiable_mesh_attributes[i] for i in selected_mesh_attribute_indices
            ]

            # Select modifiable non-Mesh attributes
            modifiable_non_mesh_attributes = [
                modifiable_attribute
                for modifiable_attribute in modifiable_attributes
                if modifiable_attribute[0] != mesh_component_idx
            ]
            num_selected_non_mesh_attributes = (
                num_attributes_to_modify - num_selected_mesh_attributes
            )
            selected_non_mesh_attribute_indices = np.random.choice(
                len(modifiable_non_mesh_attributes),
                num_selected_non_mesh_attributes,
                replace=False,
            )
            selected_attributes += [
                modifiable_non_mesh_attributes[i]
                for i in selected_non_mesh_attribute_indices
            ]
            return selected_attributes
        else:
            idx = np.random.choice(
                len(modifiable_attributes), num_attributes_to_modify, replace=False
            )
            return [modifiable_attributes[i] for i in idx]
    else:
        return modifiable_attributes


def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for I-RAVEN")
    main_arg_parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="number of samples for each component configuration",
    )
    main_arg_parser.add_argument(
        "--save-dir",
        type=str,
        default="~/datasets/I-RAVEN",
        help="path to folder where the generated dataset will be saved.",
    )
    main_arg_parser.add_argument(
        "--seed", type=int, default=-1, help="random seed for dataset generation"
    )
    main_arg_parser.add_argument(
        "--fuse", type=int, default=0, help="whether to fuse different configurations"
    )
    main_arg_parser.add_argument(
        "--val",
        type=float,
        default=2,
        help="the proportion of the size of validation set",
    )
    main_arg_parser.add_argument(
        "--test", type=float, default=2, help="the proportion of the size of test set"
    )
    main_arg_parser.add_argument("--position", action="store_false")
    main_arg_parser.add_argument("--color", action="store_false")
    main_arg_parser.add_argument("--type", action="store_false")
    main_arg_parser.add_argument("--size", action="store_false")
    main_arg_parser.add_argument(
        "--mesh", type=int, default=0, help="0 - no mesh, 1 - random, 2 - rules"
    )
    main_arg_parser.add_argument(
        "--configurations",
        type=str,
        default="center_single,distribute_four,distribute_nine,left_center_single_right_center_single,up_center_single_down_center_single,in_center_single_out_center_single,in_distribute_four_out_center_single",
    )
    args = main_arg_parser.parse_args()

    all_configs = {
        "center_single": build_center_single(args.mesh == 2),
        "distribute_four": build_distribute_four(args.mesh == 2),
        "distribute_nine": build_distribute_nine(args.mesh == 2),
        "left_center_single_right_center_single": build_left_center_single_right_center_single(
            args.mesh == 2
        ),
        "up_center_single_down_center_single": build_up_center_single_down_center_single(
            args.mesh == 2
        ),
        "in_center_single_out_center_single": build_in_center_single_out_center_single(
            args.mesh == 2
        ),
        "in_distribute_four_out_center_single": build_in_distribute_four_out_center_single(
            args.mesh == 2
        ),
    }
    all_configs = {
        k: v for k, v in all_configs.items() if k in args.configurations.split(",")
    }

    if args.seed == -1:
        args.seed = random.randint(1, 4231)

    if not os.path.exists(args.save_dir):
        os.mkdir(args.save_dir)
    if not args.fuse:
        for key in all_configs.keys():
            if not os.path.exists(os.path.join(args.save_dir, key)):
                os.mkdir(os.path.join(args.save_dir, key))
        separate(args, all_configs)


if __name__ == "__main__":
    main()
