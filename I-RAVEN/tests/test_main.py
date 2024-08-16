import pytest

from main import main, make_parser


@pytest.mark.parametrize(
    "params",
    [
        "--color",
        "--size",
        "--type",
        "--position",
        "--color --size",
        "--color --type",
        "--color --position",
        "--size --type",
        "--size --position",
        "--type --position",
        "--color --color-train-set-rule Progression",
        "--color --color-train-set-rule Arithmetic",
        "--color --color-train-set-rule Distribute_Three",
        "--size --size-train-set-rule Progression",
        "--size --size-train-set-rule Arithmetic",
        "--size --size-train-set-rule Distribute_Three",
        "--type --type-train-set-rule Progression",
        # "--type --type-train-set-rule Arithmetic",  # Arithmetic on Type is unsupported
        "--type --type-train-set-rule Distribute_Three",
        "--position --position-train-set-rule Progression",
        "--position --position-train-set-rule Arithmetic",
        "--position --position-train-set-rule Distribute_Three",
    ],
)
def test_separate(params: str):
    main_arg_parser = make_parser()
    args = [
        "--save-dir",
        f"/tmp/test_separate-{params.replace(' ', '')}",
        "--seed",
        "42",
        "--mesh",
        "0",
        "--num-samples",
        "10",
    ]
    args = main_arg_parser.parse_args(args + params.split(" "))
    accs = main(args)
    for config, acc in accs.items():
        assert (
            acc == 1.0
        ), f"Accuracy for configuration {config} is below 100%: {acc * 100}"
