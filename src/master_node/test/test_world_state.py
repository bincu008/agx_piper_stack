"""Tests for WorldStateManager object tracking and stable identity."""

from master_node.world_state import WorldStateManager

import pytest


def test_object_insert_and_update_retains_minimal_schema():
    """Verify upserting an object preserves the minimal 4-key schema."""
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
    """Verify positions with invalid dimensions or non-finite values fail."""
    state = WorldStateManager()

    with pytest.raises(ValueError):
        state.upsert_object('obj_001', 'apple', [0.1, 0.2], 'target', True)


def test_observation_association_and_conservative_removal():
    """Verify spatial association and multi-miss removal behavior."""
    state = WorldStateManager()
    first = state.associate_observation('brown cup', [0.4, 0.1, 0.2])
    second = state.associate_observation('brown cup', [0.43, 0.1, 0.2])

    assert first == second
    assert state.record_expected_miss('brown cup') == []
    assert state.record_expected_miss('brown cup') == []
    assert state.record_expected_miss('brown cup') == [first]
    assert state.objects_named('brown cup') == []


def test_repeated_observations_retain_id():
    """Verify repeated observations of same physical object reuse ID."""
    state = WorldStateManager()
    first = state.associate_observation('beige cup', [0.4, 0.1, 0.2])
    second = state.associate_observation('beige cup', [0.41, 0.1, 0.2])
    third = state.associate_observation('beige cup', [0.415, 0.105, 0.2])

    assert first == second == third == 'obj_001'
    obj = state.get_object('obj_001')
    assert obj['name'] == 'beige cup'
    assert obj['position'] == [0.415, 0.105, 0.2]
    assert len(state.snapshot()['objects']) == 1


def test_case_and_spacing_variations_retain_id():
    """Verify case and whitespace variations map to the same object ID."""
    state = WorldStateManager()
    first = state.associate_observation('beige cup', [0.4, 0.1, 0.2])
    second = state.associate_observation(
        '   BEIGE   cup   ', [0.42, 0.1, 0.2]
    )
    third = state.associate_observation('Beige Cup', [0.39, 0.1, 0.2])

    assert first == second == third == 'obj_001'
    obj = state.get_object('obj_001')
    assert obj['name'] == 'beige cup'
    aliases = state.get_aliases('obj_001')
    assert 'beige cup' in [a.casefold() for a in aliases]
    assert len(state.snapshot()['objects']) == 1


def test_explicit_body_rim_instruction_suffixes_retain_id():
    """Verify trailing grasp instructions are separated and retain same ID."""
    state = WorldStateManager()
    first = state.associate_observation('beige cup', [0.4, 0.1, 0.2])
    second = state.associate_observation(
        'beige cup, grasp its body', [0.41, 0.1, 0.2]
    )
    third = state.associate_observation(
        'beige cup, grasp its rim', [0.42, 0.1, 0.2]
    )
    fourth = state.associate_observation(
        'beige cup - grasp the handle', [0.41, 0.105, 0.2]
    )
    fifth = state.associate_observation(
        'beige cup (grasp its lip)', [0.405, 0.1, 0.2]
    )

    assert first == second == third == fourth == fifth == 'obj_001'
    obj = state.get_object('obj_001')
    assert obj['name'] == 'beige cup'
    aliases = state.get_aliases('obj_001')
    assert 'beige cup' in aliases
    assert 'beige cup, grasp its body' in aliases
    assert 'beige cup, grasp its rim' in aliases
    meta = state.get_object_metadata('obj_001')
    assert meta['last_instruction'] == 'grasp its lip'
    assert len(state.snapshot()['objects']) == 1


def test_two_spatially_distinct_bowls_remain_separate():
    """Verify identical names at different positions stay distinct objects."""
    state = WorldStateManager()
    bowl1 = state.associate_observation('white bowl', [0.2, 0.1, 0.1])
    bowl2 = state.associate_observation('white bowl', [0.5, 0.4, 0.1])

    assert bowl1 != bowl2
    assert bowl1 == 'obj_001'
    assert bowl2 == 'obj_002'

    # Observations near each bowl update only that bowl
    b1_updated = state.associate_observation(
        'white bowl, grasp its rim', [0.21, 0.1, 0.1]
    )
    b2_updated = state.associate_observation(
        'white bowl, grasp its body', [0.49, 0.4, 0.1]
    )

    assert b1_updated == bowl1
    assert b2_updated == bowl2
    assert len(state.snapshot()['objects']) == 2

    # Querying by semantic name returns both distinct objects
    named = state.objects_named('white bowl')
    assert len(named) == 2
    assert {id_ for id_, _ in named} == {bowl1, bowl2}


def test_meaningful_descriptors_are_retained_and_not_conflated():
    """Verify descriptors like small/large/left/right prevent false match."""
    state = WorldStateManager()
    small_id = state.associate_observation(
        'small white bowl', [0.2, 0.1, 0.1]
    )
    # Nearby object but with a different meaningful descriptor:
    large_id = state.associate_observation(
        'large white bowl', [0.22, 0.1, 0.1]
    )
    assert small_id != large_id

    left_id = state.associate_observation('cup on left', [0.1, 0.3, 0.2])
    right_id = state.associate_observation('cup on right', [0.12, 0.3, 0.2])
    assert left_id != right_id

    # Observation with instruction matches the correct descriptor
    small_again = state.associate_observation(
        'small white bowl, grasp its rim', [0.205, 0.1, 0.1]
    )
    assert small_again == small_id
    assert state.get_object(small_id)['name'] == 'small white bowl'


def test_existing_held_and_placed_state_survives_observation_updates():
    """Verify held and placed roles are not overwritten by new observations."""
    state = WorldStateManager()

    # Case 1: Held object
    cup_id = state.associate_observation('beige cup', [0.4, 0.1, 0.2])
    state.set_holding(cup_id)
    assert state.is_held(cup_id)
    assert state.get_object(cup_id)['role'] == 'held'
    assert state.snapshot()['robot']['holding'] == cup_id

    # New observation while held must not overwrite role to 'target'
    cup_again = state.associate_observation(
        'beige cup, grasp its body', [0.41, 0.1, 0.2]
    )
    assert cup_again == cup_id
    assert state.is_held(cup_id)
    assert state.get_object(cup_id)['role'] == 'held'
    assert state.snapshot()['robot']['holding'] == cup_id

    # Case 2: Placed object
    bowl_id = state.associate_observation('white bowl', [0.3, 0.3, 0.1])
    state.upsert_object(
        bowl_id, 'white bowl', [0.3, 0.3, 0.1], role='placed', visible=True
    )
    assert state.is_placed(bowl_id)
    assert state.get_object(bowl_id)['role'] == 'placed'

    # New observation arriving at placed location must not overwrite role
    bowl_again = state.associate_observation(
        'white bowl, grasp its rim', [0.31, 0.3, 0.1]
    )
    assert bowl_again == bowl_id
    assert state.is_placed(bowl_id)
    assert state.get_object(bowl_id)['role'] == 'placed'


def test_held_object_protected_from_expected_miss_removal():
    """Verify objects currently held are not removed by expected misses."""
    state = WorldStateManager()
    cup_id = state.associate_observation('brown cup', [0.4, 0.1, 0.2])
    state.set_holding(cup_id)

    # Calling expected miss on held object does not remove it
    for _ in range(5):
        assert state.record_expected_miss('brown cup') == []
    assert state.get_object(cup_id) is not None
    assert state.is_held(cup_id)


def test_memory_upsert_preserves_canonical_name_and_placed_state():
    """The production memory path upserts role object after association."""
    state = WorldStateManager()
    obj = state.associate_observation('beige cup, grasp its body', [.4, .1, .2])
    state.upsert_object(obj, 'beige cup', [.4, .1, .2], 'placed', True)
    state.associate_observation('beige cup, grasp its rim', [.401, .1, .2])
    state.upsert_object(obj, 'beige cup, grasp its rim', [.401, .1, .2], 'object', True)
    assert state.get_object(obj)['name'] == 'beige cup'
    assert state.is_placed(obj)
    assert 'beige cup, grasp its rim' in state.get_aliases(obj)


def test_clearing_holding_does_not_leave_a_phantom_attachment():
    """An explicit empty-hand update cannot leave the old role held forever."""
    state = WorldStateManager()
    obj = state.associate_observation('cup', [.4, .1, .2])
    state.set_holding(obj)
    state.set_holding(None)
    assert not state.is_held(obj)
    assert state.get_object(obj)['role'] != 'placed'  # No placement is inferred.
    assert state.snapshot()['robot']['holding'] is None


def test_generated_identity_does_not_overwrite_explicit_object_id():
    """Public upsert IDs and generated IDs must coexist without data loss."""
    state = WorldStateManager()
    state.upsert_object('obj_001', 'cup', [.1, .1, .1], 'target', True)
    bowl = state.associate_observation('bowl', [.5, .1, .1])
    assert bowl != 'obj_001'
    assert state.get_object('obj_001')['name'] == 'cup'


@pytest.mark.parametrize('name', [' ', '\t\n'])
def test_whitespace_only_upsert_name_is_rejected(name):
    state = WorldStateManager()
    with pytest.raises(ValueError):
        state.upsert_object('one', name, [.1, .2, .3], 'target', True)
    assert not state.snapshot()['objects']


@pytest.mark.parametrize('tolerance', [float('nan'), float('inf'), -.01])
def test_invalid_association_tolerance_cannot_merge_or_duplicate_objects(tolerance):
    state = WorldStateManager()
    with pytest.raises(ValueError):
        state.associate_observation('cup', [.1, .2, .3], tolerance=tolerance)
    assert not state.snapshot()['objects']


def test_ordinary_object_descriptor_is_not_a_bare_grasp_instruction():
    from master_node.world_state import parse_object_description
    assert parse_object_description('cup with grasp handle') == ('cup with grasp handle', None)
