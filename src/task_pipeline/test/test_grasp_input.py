"""Explicit contact intent is shared by image and layout recipes."""
from task_pipeline.grasp_input import affordance_from_description


def test_contact_intent():
    assert affordance_from_description('white bowl') == 'AUTO'
    assert affordance_from_description('upper RIM of bowl') == 'RIM'
    assert affordance_from_description('outer edge of plate') == 'RIM'
    assert affordance_from_description('cup handle') == 'HANDLE'
    assert affordance_from_description('cup body') == 'BODY'
