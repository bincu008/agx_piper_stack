"""Shared placement search and grasp/goal validation policy."""

import copy
import time


def search_placements(search, *args, timing_sink=None, **kwargs):
    """Use the same height-map engine for every input adapter."""
    started = time.monotonic()
    try:
        result = search(*args, **kwargs)
    except Exception:
        if timing_sink:
            timing_sink(time.monotonic() - started, False, 'SEARCH_FAILED')
        raise
    if timing_sink:
        timing_sink(time.monotonic() - started, bool(result[0]),
                    'CANDIDATES_READY' if result[0] else 'NO_STABLE_PACKING_CANDIDATE')
    return result


def validate_placements(request, call_backend, check_cancel=lambda: None, timing_sink=None):
    """Validate through one shared policy and measure exceptional exits too."""
    started = time.monotonic()
    try:
        return _validate_placements(request, call_backend, check_cancel, timing_sink)
    except Exception as error:
        if timing_sink:
            timing_sink(time.monotonic() - started, False, str(error))
        raise


def _validate_placements(request, call_backend, check_cancel, timing_sink):
    """Stop on full success, preserving alternatives after motion failures."""
    from my_actions.srv import PlanPlacement

    started = time.monotonic()
    candidates = list(request.candidates)
    grasps = [request.grasp_candidate, *request.alternative_grasps]
    batch_size = request.validation_batch_size or max(1, len(candidates))
    blocked = set(request.blocked_lift_grasp_ids)
    calls = []
    submitted = 0
    result = PlanPlacement.Response(status='NO_LIFT_FEASIBLE_GRASP')
    for index in range(0, len(candidates), batch_size):
        pending = [(candidates[index:index + batch_size], grasps)]
        while pending:
            check_cancel()
            goals, active = pending.pop(0)
            if request.retry_motion_failures:
                active = [g for g in active if g.id not in blocked]
            if not active:
                continue
            query = copy.copy(request)
            query.candidates = goals
            query.grasp_candidate = active[0]
            query.alternative_grasps = active[1:]
            # Downstream API dispatch is a single backend operation. The outer
            # shared policy owns batching/retries; never apply it twice.
            query.validation_batch_size = 0
            query.retry_motion_failures = False
            query.blocked_lift_grasp_ids = []
            submitted += len(goals)
            result = call_backend(query)
            check_cancel()
            # Never accept a payload belonging to another candidate or grasp.
            if result.success or (request.retry_motion_failures and result.motion_status):
                if (result.selected_candidate.id not in {g.id for g in goals} or
                        result.selected_grasp_candidate.id not in {g.id for g in active}):
                    result.success = False
                    result.status = 'PLACEMENT_RESPONSE_ASSOCIATION_MISMATCH'
                    result.motion_status = ''
                elif (result.success and request.plan_pick_place_motion and
                      not result.motion_success):
                    result.success = False
                    result.status = 'PLACEMENT_MOTION_NOT_VALIDATED'
            calls.append({
                'success': result.success, 'status': result.status,
                'motion_status': result.motion_status,
                'grasp_id': result.selected_grasp_candidate.id,
                'candidate_id': result.selected_candidate.id,
                'pose_validation_seconds': result.pose_validation_seconds,
                'descent_search_seconds': result.descent_search_seconds,
                'lift_seconds': result.lift_seconds,
                'transfer_seconds': result.transfer_seconds,
            })
            if result.success or not request.retry_motion_failures:
                break
            if not result.motion_status:
                continue
            rejected = result.selected_grasp_candidate.id
            remaining = [g for g in active if g.id != rejected]
            if remaining:
                pending.insert(0, (goals, remaining))
            if result.motion_status.startswith('LIFT_'):
                blocked.add(rejected)
            else:
                other_goals = [g for g in goals if g.id != result.selected_candidate.id]
                if other_goals:
                    pending.append((other_goals, [g for g in active if g.id == rejected]))
        if result.success:
            break
    result.validation_seconds = time.monotonic() - started
    result.validated_candidate_count = submitted
    result.blocked_lift_grasp_ids = sorted(blocked)
    if not result.success:
        motion_failures = list(dict.fromkeys(
            call['motion_status'] for call in calls
            if not call['success'] and call['motion_status']))
        if motion_failures and not result.motion_status:
            # A later IK-only rejection must not hide why an earlier feasible
            # placement could not complete its held-object motion.
            result.status += '; earlier_motion_failures=' + '|'.join(
                failure[:240] for failure in motion_failures[:6])
    import json
    result.validation_calls_json = json.dumps(calls)
    if timing_sink:
        timing_sink(result.validation_seconds, result.success, result.status)
    return result
