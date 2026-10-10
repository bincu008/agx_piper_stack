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
