"""范围引用仅枚举已归档图件的实际报告编号。"""
from copy import deepcopy

import pytest

from reporting import analysis_pdf as pdf


def test_range_uses_actual_nonconsecutive_report_numbers():
    numbers = {"F01": 9, "F02": 2, "F03": 17, "F04": 5, "F05": 12}
    before = deepcopy(numbers)
    assert pdf._replace_refs("见[F01-F05]。", numbers) == "见图9、图2、图17、图5、图12。"
    assert numbers == before


@pytest.mark.parametrize("value", ["[F01-F03]", "[F01 - F03]", "[F01\t-\tF03]"])
def test_optional_simple_spaces(value):
    assert pdf._replace_refs(value, {"F01": 1, "F02": 4, "F03": 7}) == "图1、图4、图7"


@pytest.mark.parametrize("numbers", [{"F01": 1, "F03": 7}, {"F02": 4, "F03": 7}, {"F01": 1, "F02": 4}, {}])
def test_missing_middle_or_endpoint_never_makes_up_numbers(numbers):
    assert pdf._replace_refs("见[F01-F03]", numbers) == "见未归档图件"


@pytest.mark.parametrize("value", ["[F03-F01]", "[F01-F01]", "[F00-F02]", "[F1-F3]", "[F001-F003]"])
def test_reverse_equal_zero_and_noncanonical_ids_are_not_inferred(value):
    assert pdf._replace_refs(value, {"F00": 4, "F01": 1, "F02": 2, "F03": 3}) == "未归档图件"


@pytest.mark.parametrize("value", ["[F01-F999999]", "[F01-F1000000]", "[F01-F" + "9" * 5000 + "]"])
def test_huge_range_rejected_before_iteration_or_integer_conversion(value):
    class NeverLookup(dict):
        def __len__(self):
            return 10000000
        def __contains__(self, key):
            raise AssertionError("超大范围不应进行图件逐项查询")
    assert pdf._replace_refs(value, NeverLookup()) == "未归档图件"


def test_range_limit_is_bounded_even_when_mapping_is_large():
    numbers = {f"F{i:02d}": i for i in range(1, 258)}
    assert pdf._replace_refs("[F01-F257]", numbers) == "未归档图件"
    assert pdf._replace_refs("[F01-F256]", numbers).count("图") == 256


def test_number_width_can_naturally_cross_99():
    assert pdf._replace_refs("[F99-F101]", {"F99": 4, "F100": 8, "F101": 3}) == "图4、图8、图3"


def test_single_ids_keep_existing_behavior_with_mixed_range():
    assert pdf._replace_refs("[F01]、[F01-F02]、[F09]", {"F01": 6, "F02": 3}) == "图6、图6、图3、未归档图件"


@pytest.mark.parametrize("value", ["普通文字1-5不改", "F01-F05", "[E01-E05]", "[F01/F05]", "[F01–F05]", "[F01\n-F05]"])
def test_plain_text_and_out_of_scope_syntax_remain_unchanged(value):
    assert pdf._replace_refs(value, {"F01": 1, "F05": 2}) == value
