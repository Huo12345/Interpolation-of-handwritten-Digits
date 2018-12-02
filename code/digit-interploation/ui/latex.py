from typing import List


def create_confusion_table_matrix(filename: str, data: List[List[int]], labels: List[str], caption: str = None,
                                  label: str = None):
    with open(filename, "w") as text_file:
        text_file.write("\\documentclass[Interploate_hadwritten_Digits.tex]{subfiles}")
        text_file.write("\\begin{document}")
        text_file.write("\\begin{table}[H]")
        text_file.write("\\centering")
        text_file.write("\\begin{tabular}{|%s|}" % "|".join(['r' for _ in range(len(labels) + 1)]))
        text_file.write("\\hline")
        text_file.write("Actual \\textbackslash Prediction & %s \\\\ \\hline" % " & ".join(labels))
        for i in range(len(data)):
            text_file.write("%s & %s \\\\ \\hline" % (labels[i], " & ".join(map(str, data[i]))))
        text_file.write("\\end{tabular}")
        if caption is not None:
            text_file.write("\\caption{%s}" % caption)
        if label is not None:
            text_file.write("\\label{%s}" % label)
        text_file.write("\\end{table}")
        text_file.write("\\end{document}")
