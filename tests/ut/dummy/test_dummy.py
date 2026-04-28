from tests.ut.conftest import npu_test  # noqa E402


@npu_test(num_npus=1, npu_type="a2")
def test_dummy():
    assert True


@npu_test(num_npus=2, npu_type="a3")
def test_dummy_with_a3():
    assert True


def test_dummy_without_npu():
    assert True
