def batch(iterable, n=1):
    """Utility function to batch sentences."""
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]
