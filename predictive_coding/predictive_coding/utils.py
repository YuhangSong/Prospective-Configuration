import warnings


def _is_positive_int(x):
    return (isinstance(x, int) and x > 0)


def slow_down_warning(base, property, solution):
    warnings.warn(
        (
            "In {}, you have {} enabled, this will slow down training. Set to {} to disable it. ".format(
                base, property, solution
            )
        ),
        category=RuntimeWarning
    )
