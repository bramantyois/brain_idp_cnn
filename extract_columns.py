#!/usr/bin/env python3
#
# usage:
#   python3 extract_columns.py 21003-2.0 25008-2.0 ...

from pathlib import Path
from tqdm import tqdm
import pandas as pd
from typing import List
import argparse
import sys

UKBB_LABELS = Path("/ritter/share/data/UKBB_2020/ukb44573.csv")


def ukbb_read_csv(
    path: Path, columns: List[str], chunk_size: int = 1000
) -> pd.DataFrame:
    # Ensure EID is in columns
    if "eid" not in columns:
        columns.insert(0, "eid")
    out_columns = columns.copy()
    if "25008-2.0" not in columns:
        columns.append("25008-2.0")

    # Approximate number of lines so the we can provide a progress bar
    filesize = path.stat().st_size
    line_size = 0.0
    with path.open("r") as f:
        next(f)
        for i in range(100):
            line_size += len(f.readline()) * 0.01
    approx_num_lines = filesize // int(line_size) + 1
    print("Approx num lines:", approx_num_lines)

    # Read dataframe chunked, only for specified columns
    tmp_lst = []
    with pd.read_csv(
        path, chunksize=chunk_size, usecols=columns, low_memory=False
    ) as reader:
        t = tqdm(total=approx_num_lines)
        for chunk in reader:
            # Skip chunks based on condition
            fixed_chunk = chunk[chunk["25008-2.0"] > 0]
            if not fixed_chunk.empty:
                tmp = fixed_chunk[out_columns]
                tmp_lst.append(tmp)
            t.update(chunk_size)

    # Create dataframe, set index to eid for faster querying
    df = pd.concat(tmp_lst)
    df.set_index("eid", inplace=True)
    df.sort_index(inplace=True)
    return df


if __name__ == "__main__":
    columns = sys.argv[1:]
    df = ukbb_read_csv(UKBB_LABELS, columns)
    df.to_csv("out.csv")