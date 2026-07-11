def to_float_list(value) -> list[float]:
    """
    Convert pgvector/numpy/list-like vector values from DB rows into plain floats.

    pgvector.psycopg2 can return a Vector object that exposes to_list()/to_numpy()
    but is not directly iterable, so all DB-boundary vector reads should pass
    through this helper before vector math or JSON serialization.
    """
    if hasattr(value, "to_list"):
        value = value.to_list()
    elif hasattr(value, "to_numpy"):
        value = value.to_numpy()
    return [float(x) for x in value]
