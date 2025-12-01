import os
from itertools import zip_longest

root="data"
dirs = sorted(os.listdir(root))

rows = []
for d in dirs:
    path = os.path.join(root,d)
    if os.path.isdir(path):
        files = os.listdir(path)
        rows.append([d] + files)

# Pad columns to equal length
max_len = max(len(r) for r in rows)
rows = [r + [""]*(max_len-len(r)) for r in rows]

# Print side-by-side column tree
print(f"\n{' '*30}{root}\n")
for i,row in enumerate(rows):
    col_name = row[0].ljust(20)
    branch = "├─ " if i < len(rows)-1 else "└─ "
    print(f"{branch}{col_name}", end="")

    for f in row[1:]:
        if f!="":
            print(f"{f:>18}", end="")
    print()
