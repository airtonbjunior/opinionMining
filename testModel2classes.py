# Airton Bordin Junior
# airtonbjunior@gmail.com
# Federal University of Goias (UFG)
# Computer Science Master's Degree

import time

from variables import *
from functions import *

# log time
start = time.time()

if __name__ == "__main__":
    print("[starting test module (2 classes)]")
    getDictionary("test")

    functions_to_evaluate = []

    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename()

    with open(file_path, 'r') as inF:
        for line in inF:
            if not line.startswith("[") and line not in ('\n', '\r\n') and not line.startswith("#"):
                functions_to_evaluate.append(str(line))

    for function_to_evaluate in functions_to_evaluate:
        evaluateMessages_2classes(function_to_evaluate)

    resultsAnalysis_2classes()

# log time
end = time.time()
print("\n\n[script ends after " + str(format(end - start, '.3g')) + " seconds]")