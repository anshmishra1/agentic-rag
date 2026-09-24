"""Deprecated compatibility entry point for hybrid index creation."""

import warnings

from create_hybrid_index import main


if __name__ == "__main__":
    warnings.warn(
        "scripts/create_index.py is deprecated; use "
        "scripts/create_hybrid_index.py instead.",
        DeprecationWarning,
        stacklevel=1,
    )
    main()
