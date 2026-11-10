"""Shared grasp/preparation cache lifetime (600s TTL); active execution has its own watchdog."""

GRASP_CACHE_TTL_SECONDS = 10 * 60  # 600 seconds as specified in unified workspace proposal


def grasp_cache_ttl(request=None):
    """Use the default 600s lifetime or a caller's shorter explicit lifetime."""
    if request is None:
        return float(GRASP_CACHE_TTL_SECONDS)
    explicit = getattr(request, 'ttl_seconds', 0)
    return min(float(GRASP_CACHE_TTL_SECONDS), float(explicit or GRASP_CACHE_TTL_SECONDS))
