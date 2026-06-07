from __future__ import annotations

import pandas as pd


def has_data(frame: pd.DataFrame) -> bool:
    return isinstance(frame, pd.DataFrame) and not frame.empty
