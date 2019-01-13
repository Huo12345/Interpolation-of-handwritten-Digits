

for i in range(10):
    for j in range(i + 1, 10):
        for k in [25, 50, 75]:
            id = "%d_%d_%d" % (i, j, k)
            print("%d -> %d: %d \\%% & \\includegraphics[scale=0.25]{img/results/interpolated_%s_inverted.png} & "
                  "\\includegraphics[scale=0.25]{img/results/interpolated_%s_squared_inverted.png} "
                  "& \\includegraphics[scale=0.25]{img/results/interpolated_%s_approximated.png} \\\\" %
                  (i, j, k, id, id, id))
