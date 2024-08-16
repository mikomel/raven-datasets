import pytest

from sampling import do_enforce_train_set_rule


# fmt: off
@pytest.mark.parametrize(
    "split,configuration,component_idx,ood_attribute,train_set_rule,expected",
    [
        # === Position + Constant ===
        # Position must be Constant in Center irrespectively of the dataset split
        # This also applies to Left-Right, Up-Down, Out-InCenter
        ("train", "center_single", 0, "Position", "Constant", True),
        ("test", "center_single", 0, "Position", "Constant", True),
        # Position may be not Constant in 2x2 in the test split
        # This also applies to 3x3 and the Inner component of Out-InGrid
        ("train", "distribute_four", 0, "Position", "Constant", True),
        ("test", "distribute_four", 0, "Position", "Constant", False),
        ("train", "in_distribute_four_out_center_single", 0, "Position", "Constant", True),
        ("test", "in_distribute_four_out_center_single", 0, "Position", "Constant", True),
        ("train", "in_distribute_four_out_center_single", 1, "Position", "Constant", True),
        ("test", "in_distribute_four_out_center_single", 1, "Position", "Constant", False),
        # === Position + not Constant ===
        # Position can't be modified (i.e. must be Constant) in Center irrespectively of the dataset split
        # This also applies to Left-Right, Up-Down, Out-InCenter
        ("train", "center_single", 0, "Position", "Progression", False),
        ("test", "center_single", 0, "Position", "Progression", False),
        # Position may be modified in 2x2 in the train split
        # This also applies to 3x3 and the Inner component of Out-InGrid
        ("train", "distribute_four", 0, "Position", "Progression", True),
        ("test", "distribute_four", 0, "Position", "Progression", False),
        ("train", "in_distribute_four_out_center_single", 0, "Position", "Progression", False),
        ("test", "in_distribute_four_out_center_single", 0, "Position", "Progression", False),
        ("train", "in_distribute_four_out_center_single", 1, "Position", "Progression", True),
        ("test", "in_distribute_four_out_center_single", 1, "Position", "Progression", False),
        # === Type ===
        # Type rule may be always enforced in the train split
        ("train", "center_single", 0, "Type", "Constant", True),
        ("test", "center_single", 0, "Type", "Constant", False),
        ("train", "distribute_four", 0, "Type", "Constant", True),
        ("test", "distribute_four", 0, "Type", "Constant", False),
        ("train", "in_distribute_four_out_center_single", 0, "Type", "Constant", True),
        ("test", "in_distribute_four_out_center_single", 0, "Type", "Constant", False),
        ("train", "in_distribute_four_out_center_single", 1, "Type", "Constant", True),
        ("test", "in_distribute_four_out_center_single", 1, "Type", "Constant", False),
        ("train", "center_single", 0, "Type", "Progression", True),
        ("test", "center_single", 0, "Type", "Progression", False),
        ("train", "distribute_four", 0, "Type", "Progression", True),
        ("test", "distribute_four", 0, "Type", "Progression", False),
        ("train", "in_distribute_four_out_center_single", 0, "Type", "Progression", True),
        ("test", "in_distribute_four_out_center_single", 0, "Type", "Progression", False),
        ("train", "in_distribute_four_out_center_single", 1, "Type", "Progression", True),
        ("test", "in_distribute_four_out_center_single", 1, "Type", "Progression", False),
        # === Size ===
        # Size rule may be always enforced in the train split
        ("train", "center_single", 0, "Size", "Constant", True),
        ("test", "center_single", 0, "Size", "Constant", False),
        ("train", "distribute_four", 0, "Size", "Constant", True),
        ("test", "distribute_four", 0, "Size", "Constant", False),
        ("train", "in_distribute_four_out_center_single", 0, "Size", "Constant", True),
        ("test", "in_distribute_four_out_center_single", 0, "Size", "Constant", False),
        ("train", "in_distribute_four_out_center_single", 1, "Size", "Constant", True),
        ("test", "in_distribute_four_out_center_single", 1, "Size", "Constant", False),
        ("train", "center_single", 0, "Size", "Progression", True),
        ("test", "center_single", 0, "Size", "Progression", False),
        ("train", "distribute_four", 0, "Size", "Progression", True),
        ("test", "distribute_four", 0, "Size", "Progression", False),
        ("train", "in_distribute_four_out_center_single", 0, "Size", "Progression", True),
        ("test", "in_distribute_four_out_center_single", 0, "Size", "Progression", False),
        ("train", "in_distribute_four_out_center_single", 1, "Size", "Progression", True),
        ("test", "in_distribute_four_out_center_single", 1, "Size", "Progression", False),
        # Arithmetic on Size can't be applied to the Outer component in Out-In configurations
        ("train", "in_distribute_four_out_center_single", 0, "Size", "Arithmetic", False),
        ("test", "in_distribute_four_out_center_single", 0, "Size", "Arithmetic", False),
        # === Color ===
        ("train", "center_single", 0, "Color", "Constant", True),
        ("test", "center_single", 0, "Color", "Constant", False),
        ("train", "distribute_four", 0, "Color", "Constant", True),
        ("test", "distribute_four", 0, "Color", "Constant", False),
        # Color must be Constant in the Outer component of Out-InCenter and Out-InGrid irrespectively of the dataset split
        ("train", "in_distribute_four_out_center_single", 0, "Color", "Constant", True),
        ("test", "in_distribute_four_out_center_single", 0, "Color", "Constant", True),
        ("train", "in_distribute_four_out_center_single", 1, "Color", "Constant", True),
        ("test", "in_distribute_four_out_center_single", 1, "Color", "Constant", False),
        # === Color + not Constant ===
        ("train", "center_single", 0, "Color", "Progression", True),
        ("test", "center_single", 0, "Color", "Progression", False),
        ("train", "distribute_four", 0, "Color", "Progression", True),
        ("test", "distribute_four", 0, "Color", "Progression", False),
        # Color can't be modified (i.e. must be Constant) in the Outer component
        # of Out-InCenter and Out-InGrid irrespectively of the dataset split
        ("train", "in_distribute_four_out_center_single", 0, "Color", "Progression", False),
        ("test", "in_distribute_four_out_center_single", 0, "Color", "Progression", False),
        ("train", "in_distribute_four_out_center_single", 1, "Color", "Progression", True),
        ("test", "in_distribute_four_out_center_single", 1, "Color", "Progression", False),
    ],
)
# fmt: on
def test_do_enforce_train_set_rule(
    split: str,
    configuration: str,
    component_idx: int,
    ood_attribute: str,
    train_set_rule: str,
    expected: bool,
):
    ood_attribute_idx = ["Position", "Type", "Size", "Color"].index(ood_attribute)
    actual = do_enforce_train_set_rule(
        split, configuration, component_idx, ood_attribute_idx, train_set_rule
    )
    assert expected == actual
