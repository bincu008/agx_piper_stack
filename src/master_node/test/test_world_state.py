from master_node.world_state import WorldStateManager
import pytest


def test_object_insert_and_update_retains_minimal_schema():
    state = WorldStateManager()

    state.upsert_object(
        'obj_001',
        name='apple',
        position=[0.4, 0.1, 0.2],
        role='target',
        visible=True,
    )
    state.upsert_object(
        'obj_001',
        name='apple',
        position=[0.42, 0.08, 0.21],
        role='target',
        visible=False,
    )

    snapshot = state.snapshot()
    assert snapshot['objects']['obj_001'] == {
        'name': 'apple',
        'position': [0.42, 0.08, 0.21],
        'role': 'target',
        'visible': False,
    }
    assert snapshot['robot'] == {'state': 'IDLE', 'holding': None}


def test_invalid_world_position_is_rejected():
    state = WorldStateManager()

    with pytest.raises(ValueError):
        state.upsert_object('obj_001', 'apple', [0.1, 0.2], 'target', True)


def test_observation_association_and_conservative_removal():
    state = WorldStateManager()
    first = state.associate_observation('brown cup', [0.4, 0.1, 0.2])
    second = state.associate_observation('brown cup', [0.43, 0.1, 0.2])

    assert first == second
    assert state.record_expected_miss('brown cup') == []
    assert state.record_expected_miss('brown cup') == []
    assert state.record_expected_miss('brown cup') == [first]
    assert state.objects_named('brown cup') == []
