import numpy as np
from typing import List


def create_confusion_table_matrix(filename: str, data: np.ndarray, labels: List[str], caption: str = None,
                                  label: str = None):
    with open(filename, "w") as text_file:
        text_file.write("\\documentclass[Interploate_hadwritten_Digits.tex]{subfiles}\n")
        text_file.write("\\begin{document}\n")
        text_file.write("\\begin{table}[H]\n")
        text_file.write("\\centering\n")
        text_file.write("\\begin{tabular}{|%s|}\n" % "|".join(['r' for _ in range(len(labels) + 1)]))
        text_file.write("\\hline\n")
        text_file.write("Actual \\textbackslash Prediction & %s \\\\ \\hline\n" % " & ".join(labels))
        for i in range(len(data)):
            text_file.write("%s & %s \\\\ \\hline\n" % (labels[i], " & ".join(map(str, data[i]))))
        text_file.write("\\end{tabular}\n")
        if caption is not None:
            text_file.write("\\caption{%s}\n" % caption)
        if label is not None:
            text_file.write("\\label{%s}\n" % label)
        text_file.write("\\end{table}\n")
        text_file.write("\\end{document}")
    print("Created %s" % filename)
