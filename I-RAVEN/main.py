# -*- coding: utf-8 -*-


import argparse
import copy
import os
import random
import sys

import numpy as np
from tqdm import trange

from build_tree import (build_center_single, build_distribute_four,
                        build_distribute_nine,
                        build_in_center_single_out_center_single,
                        build_in_distribute_four_out_center_single,
                        build_left_center_single_right_center_single,
                        build_up_center_single_down_center_single)
from const import IMAGE_SIZE, RULE_ATTR
from rendering import (generate_matrix, generate_matrix_answer, imsave, imshow,
                       render_panel)
from Rule import Rule_Wrapper
from sampling import sample_attr, sample_attr_avail, sample_rules
from serialize import dom_problem, serialize_aot, serialize_rules, serialize_modifications
from solver import solve


def merge_component(dst_aot, src_aot, component_idx):
    src_component = src_aot.children[0].children[component_idx]
    dst_aot.children[0].children[component_idx] = src_component


def separate(args, all_configs):
    random.seed(args.seed)
    np.random.seed(args.seed)

    for key in all_configs.keys():
        acc = 0
        for k in trange(args.num_samples):
            count_num = k % 10
            if count_num < (10 - args.val - args.test):
                set_name = "train"
            elif count_num < (10 - args.test):
                set_name = "val"
            else:
                set_name = "test"

            root = all_configs[key]
            num_of_components = len(root.children[0].children)
            is_last_mesh = args.mesh == 2
            while True:
                rule_groups = sample_rules(num_of_components, is_last_mesh, args.position, args.type, args.size, args.color)
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

            imgs = [render_panel(row_1_1, args.mesh == 1),
                    render_panel(row_1_2, args.mesh == 1),
                    render_panel(row_1_3, args.mesh == 1),
                    render_panel(row_2_1, args.mesh == 1),
                    render_panel(row_2_2, args.mesh == 1),
                    render_panel(row_2_3, args.mesh == 1),
                    render_panel(row_3_1, args.mesh == 1),
                    render_panel(row_3_2, args.mesh == 1),
                    np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)]
            context = [row_1_1, row_1_2, row_1_3, row_2_1, row_2_2, row_2_3, row_3_1, row_3_2]
            modifiable_attr = sample_attr_avail(rule_groups, row_3_3)
            answer_AoT = copy.deepcopy(row_3_3)
            candidates = [answer_AoT]  

            attr_num = 3
            if attr_num <= len(modifiable_attr):
                idx = np.random.choice(len(modifiable_attr), attr_num, replace=False)   
                selected_attr = [modifiable_attr[i] for i in idx]
            else:
                selected_attr = modifiable_attr

            mode = None
            #switch attribute 'Number' for convenience
            pos = [i for i in xrange(len(selected_attr)) if selected_attr[i][1]=='Number']
            if pos:
                pos = pos[0]
                selected_attr[pos], selected_attr[-1] = selected_attr[-1], selected_attr[pos] 

                pos = [i for i in xrange(len(selected_attr)) if selected_attr[i][1]=='Position']
                if pos:
                    mode = 'Position-Number'
            values = [] 
            if len(selected_attr) >= 3:
                mode_3 = None
                if mode == 'Position-Number':
                    mode_3 = '3-Position-Number'  
                for i in xrange(attr_num):    
                    component_idx, attr_name, min_level, max_level, attr_uni = selected_attr[i][0], selected_attr[i][1], selected_attr[i][3], selected_attr[i][4], selected_attr[i][5]                
                    value = answer_AoT.sample_new_value(component_idx, attr_name, min_level, max_level, attr_uni, mode_3)
                    values.append(value)
                    tmp = []
                    for j in candidates:
                        new_AoT = copy.deepcopy(j)
                        new_AoT.apply_new_value(component_idx, attr_name, value)
                        tmp.append(new_AoT)
                    candidates += tmp   

            elif len(selected_attr) == 2:     
                component_idx, attr_name, min_level, max_level, attr_uni = selected_attr[0][0], selected_attr[0][1], selected_attr[0][3], selected_attr[0][4], selected_attr[0][5]               
                value = answer_AoT.sample_new_value(component_idx, attr_name, min_level, max_level, attr_uni, None)
                values.append(value)
                new_AoT = copy.deepcopy(answer_AoT)
                new_AoT.apply_new_value(component_idx, attr_name, value)
                candidates.append(new_AoT)
                component_idx, attr_name, min_level, max_level, attr_uni = selected_attr[1][0], selected_attr[1][1], selected_attr[1][3], selected_attr[1][4], selected_attr[1][5]
                if mode == 'Position-Number':  
                    ran,qu = 6, 1 
                else:
                    ran,qu = 3, 2   
                for i in xrange(ran):
                    value = answer_AoT.sample_new_value(component_idx, attr_name, min_level, max_level, attr_uni, None)
                    values.append(value)
                    for j in xrange(qu):
                        new_AoT = copy.deepcopy(candidates[j])
                        new_AoT.apply_new_value(component_idx, attr_name, value)
                        candidates.append(new_AoT)

            elif len(selected_attr) == 1:
                component_idx, attr_name, min_level, max_level, attr_uni = selected_attr[0][0], selected_attr[0][1], selected_attr[0][3], selected_attr[0][4], selected_attr[0][5]             
                for i in xrange(7):
                    value = answer_AoT.sample_new_value(component_idx, attr_name, min_level, max_level, attr_uni, None)
                    values.append(value)
                    new_AoT = copy.deepcopy(answer_AoT)
                    new_AoT.apply_new_value(component_idx, attr_name, value)
                    candidates.append(new_AoT) 

            random.shuffle(candidates)
            answers = []
            mods = []
            for candidate in candidates:
                answers.append(render_panel(candidate, args.mesh == 1))
                mods.append(candidate.modified_attr)




            #imsave(generate_matrix_answer(imgs + answers), "/media/dsg3/hs/RAVEN_image/experiments2/{}/{}.jpg".format(key, k))    
            
            image = imgs[0:8] + answers
            target = candidates.index(answer_AoT)
            # predicted = solve(rule_groups, context, candidates)
            is_mesh_present = start_node.children[0].children[-1].name == 'Mesh'
            max_components = len(start_node.children[0].children)
            meta_matrix, meta_target = serialize_rules(rule_groups, is_mesh_present)
            structure, meta_structure = serialize_aot(start_node)
            modifications_matrix = serialize_modifications(mods, is_mesh_present, max_components)
            np.savez("{}/{}/RAVEN_{}_{}.npz".format(args.save_dir, key, k, set_name), image=image, 
                                                                                      target=target, 
                                                                                      predict=target,
                                                                                      meta_matrix=meta_matrix,
                                                                                      meta_target=meta_target, 
                                                                                      structure=structure,
                                                                                      meta_structure=meta_structure,
                                                                                      meta_answer_mods=modifications_matrix)
            with open("{}/{}/RAVEN_{}_{}.xml".format(args.save_dir, key, k, set_name), "w") as f:
                dom = dom_problem(context + candidates, rule_groups)
                f.write(dom)
            
            # if target == predicted:
            #    acc += 1
        # print "Accuracy of {}: {}".format(key, float(acc) / args.num_samples)
    

def main():
    main_arg_parser = argparse.ArgumentParser(description="parser for I-RAVEN")
    main_arg_parser.add_argument("--num-samples", type=int, default=10000,
                                 help="number of samples for each component configuration")
    main_arg_parser.add_argument("--save-dir", type=str, default="/media/dsg3/datasets/I-RAVEN",
                                 help="path to folder where the generated dataset will be saved.")
    main_arg_parser.add_argument("--seed", type=int, default=-1,
                                 help="random seed for dataset generation")
    main_arg_parser.add_argument("--fuse", type=int, default=0,
                                 help="whether to fuse different configurations")
    main_arg_parser.add_argument("--val", type=float, default=2,
                                 help="the proportion of the size of validation set")
    main_arg_parser.add_argument("--test", type=float, default=2,
                                 help="the proportion of the size of test set")
    main_arg_parser.add_argument('--position', action='store_false')
    main_arg_parser.add_argument('--color', action='store_false')
    main_arg_parser.add_argument('--type', action='store_false')
    main_arg_parser.add_argument('--size', action='store_false')
    main_arg_parser.add_argument("--mesh", type=int, default=0,
                                 help="0 - no mesh, 1 - random, 2 - rules")
    args = main_arg_parser.parse_args()

    all_configs = {
        "center_single": build_center_single(args.mesh == 2),
        "distribute_four": build_distribute_four(args.mesh == 2),
        "distribute_nine": build_distribute_nine(args.mesh == 2),
        "left_center_single_right_center_single": build_left_center_single_right_center_single(args.mesh == 2),
        "up_center_single_down_center_single": build_up_center_single_down_center_single(args.mesh == 2),
        "in_center_single_out_center_single": build_in_center_single_out_center_single(args.mesh == 2),
        "in_distribute_four_out_center_single": build_in_distribute_four_out_center_single(args.mesh == 2)
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
