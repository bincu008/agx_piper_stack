"""Shared interpretation of explicit grasp contact descriptions."""


def affordance_from_description(description):
    """Select the requested contact region without assuming an object class."""
    target = description.lower()
    if 'handle' in target:
        return 'HANDLE'
    if any(word in target for word in ('rim', 'lip', 'edge')):
        return 'RIM'
    if any(word in target for word in ('body', 'side')):
        return 'BODY'
    return 'AUTO'


def calculate_grasps_with_recovery(request, call):
    """Retry an empty bowl AUTO result once at measured rim contacts."""
    import copy

    original = call(request)
    if (original.success or (request.affordance_type or 'AUTO').upper() != 'AUTO' or
            'bowl' not in request.target_description.casefold() or
            not original.status.startswith('NO_VALID_GEOMETRIC_GRASP')):
        return original
    query = copy.deepcopy(request)
    query.affordance_type = 'RIM'
    recovered = call(query)
    if recovered.success:
        recovered.status += ';AUTO_TO_RIM_RECOVERY'
        return recovered
    recovered.status = 'AUTO_FAILED: '+original.status+'; RIM_FAILED: '+recovered.status
    return recovered
