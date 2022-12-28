# -*- coding: utf-8 -*-


import json
import xml.etree.ElementTree as ET

import numpy as np

from const import META_STRUCTURE_FORMAT
from api import get_real_bbox, get_mask, rle_encode


def n_tree_serialize(aot):
    assert aot.is_pg
    ret = ""
    if aot.level == "Layout":
        return aot.name + "./"
    else:
        ret += aot.name + "."
        for child in aot.children:
            x = n_tree_serialize(child)
            ret += x
            ret += "."
        ret += "/"
    return ret


def serialize_aot(aot):
    """Meta Structure format
    META_STRUCTURE_FORMAT provided by const.py
    """    
    n_tree = n_tree_serialize(aot)
    meta_structure = np.zeros(len(META_STRUCTURE_FORMAT), np.uint8)
    split = n_tree.split(".")
    for node in split:
        try:
            node_index = META_STRUCTURE_FORMAT.index(node)
            meta_structure[node_index] = 1
        except ValueError:
            continue
    return split, meta_structure


def serialize_rules(rule_groups, is_mesh_present=False):
    """Meta matrix format
    ["Constant", "Progression", "Arithmetic", "Distribute_Three", "Number", "Position", "Type", "Size", "Color"]
    """
    meta_matrix = np.zeros((12, 9), np.uint8)
    counter = 0

    temp_rule_groups = rule_groups
    if is_mesh_present and len(rule_groups) != 3:
        temp_rule_groups = [rule_groups[0], [], rule_groups[1]]

    for rule_group in temp_rule_groups:
        if len(rule_group) == 0:
            counter += 4
        for rule in rule_group:
            if rule.name == "Constant":
                meta_matrix[counter, 0] = 1
            elif rule.name == "Progression":
                meta_matrix[counter, 1] = 1
            elif rule.name == "Arithmetic":
                meta_matrix[counter, 2] = 1
            else:
                meta_matrix[counter, 3] = 1
            if rule.attr == "Number/Position":
                meta_matrix[counter, 4] = 1
                meta_matrix[counter, 5] = 1
            elif rule.attr == "Number":
                meta_matrix[counter, 4] = 1
            elif rule.attr == "Position":
                meta_matrix[counter, 5] = 1
            elif rule.attr == "Type":
                meta_matrix[counter, 6] = 1
            elif rule.attr == "Size":
                meta_matrix[counter, 7] = 1
            else:
                meta_matrix[counter, 8] = 1
            counter += 1
    return meta_matrix, np.bitwise_or.reduce(meta_matrix)

def serialize_modifications(modifications, is_mesh_present=False, max_comp = 2):
    """Meta matrix format (2 rows per answer - component 0 and 1)
    ["Number", "Position", "Type", "Size", "Color"]
    """
    meta_matrix = np.zeros((24 if is_mesh_present else 16, 5), np.uint8)
    counter = 0
    for answer_mod in modifications:
        for mod in answer_mod:
            index = counter + mod[0]
            attribute = mod[1]
            if is_mesh_present and mod[0] == 1 and max_comp == 2:
                index += 1
            if attribute == "Number/Position":
                meta_matrix[index, 0] = 1
                meta_matrix[index, 1] = 1
            elif attribute == "Number":
                meta_matrix[index, 0] = 1
            elif attribute == "Position":
                meta_matrix[index, 1] = 1
            elif attribute == "Type":
                meta_matrix[index, 2] = 1
            elif attribute == "Size":
                meta_matrix[index, 3] = 1
            else:
                meta_matrix[index, 4] = 1
        counter += 3 if is_mesh_present else 2
    return meta_matrix


def dom_problem(instances, rule_groups):
    data = ET.Element("Data")
    panels = ET.SubElement(data, "Panels")
    for i in range(len(instances)):
        panel = instances[i]
        panel_i = ET.SubElement(panels, "Panel")
        struct = panel.children[0]
        struct_i = ET.SubElement(panel_i, "Struct")
        struct_i.set("name", struct.name)
        for j in range(len(struct.children)):
            component = struct.children[j]
            component_j = ET.SubElement(struct_i, "Component")
            component_j.set("id", str(j))
            component_j.set("name", component.name)
            layout = component.children[0]
            layout_k = ET.SubElement(component_j, "Layout")
            layout_k.set("name", layout.name)
            layout_k.set("Number", str(layout.number.get_value_level()))
            layout_k.set("Position", json.dumps(layout.position.values))
            layout_k.set("Uniformity", str(layout.uniformity.get_value_level()))
            for l in range(len(layout.children)):
                entity = layout.children[l]
                entity_l = ET.SubElement(layout_k, "Entity")
                entity_bbox = entity.bbox
                entity_type = entity.type.get_value()
                entity_size = entity.size.get_value()
                entity_angle = entity.angle.get_value()
                entity_l.set("bbox", json.dumps(entity_bbox))
                entity_l.set("real_bbox", json.dumps(get_real_bbox(entity_bbox, entity_type, entity_size, entity_angle)))
                entity_l.set("mask", rle_encode(get_mask(entity_bbox, entity_type, entity_size, entity_angle)))
                entity_l.set("Type", str(entity.type.get_value_level()))
                entity_l.set("Size", str(entity.size.get_value_level()))
                entity_l.set("Color", str(entity.color.get_value_level()))
                entity_l.set("Angle", str(entity.angle.get_value_level()))
    rules = ET.SubElement(data, "Rules")
    for i in range(len(rule_groups)):
        rule_group = rule_groups[i]
        rule_group_i = ET.SubElement(rules, "Rule_Group")
        rule_group_i.set("id", str(i))
        for rule in rule_group:
            rule_j = ET.SubElement(rule_group_i, "Rule")
            rule_j.set("name", rule.name)
            rule_j.set("attr", rule.attr)


    modified_attr = ET.SubElement(data, "Modified_attributes")

    for i in range(8):
        candidate = instances[i+8]

        candidate_i = ET.SubElement(modified_attr, 'Candidate')
        candidate_i.set("id", str(i))

        for attr in candidate.modified_attr:
            attr_j =  ET.SubElement(candidate_i, "Attribute")
            attr_j.set('component_id',str(attr[0]))
            attr_j.set('name',attr[1])



    return ET.tostring(data)
