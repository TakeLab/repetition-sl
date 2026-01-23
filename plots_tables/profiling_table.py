import argparse
import pandas as pd
import re

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_to_profiling", type=str, default="tables/profiling.tex")
    args = parser.parse_args()

    good_color = 'steelblue'
    bad_color = 'burntorange'
    max_value = 5

    with open(args.path_to_profiling, "r") as f:
        for line in f:
            line = line.strip()
            if line == '':
                continue

            if 'exit@' not in line:
                print(line)
                continue

            line = line.split(' & ')
            colored_line = []
            for i in line:
                if 'exit' in i:
                    colored_line.append(i)
                else:
                    value = re.match(r'\$([0-9.]+)\$', i).group(1)
                    value = float(value)

                    if value < 1:
                        color_value = (1/value - 1) / (max_value - 1)
                        color_value *= 100
                        colored_line.append(f"\\cellcolor{{{bad_color}!{round(color_value)}}}{{${value:.2f}$}}")
                    else:
                        color_value = (value - 1) / (max_value - 1)
                        color_value *= 100
                        colored_line.append(f"\\cellcolor{{{good_color}!{round(color_value)}}}{{${value:.2f}$}}")

            colored_line = ' & '.join(colored_line) + '\\\\'
            print(colored_line)

