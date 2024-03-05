import argparse
from pathlib import Path

import pandas as pd


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run file to inference using the model found in the config file")

    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input XLSX file")

    args = parser.parse_args()

    return args


def main(args):
    data = pd.read_excel(args.input)

    names = set()
    for index, row in data.iterrows():
        name = row["Bestandsnamen"]
        names.add(name)

    rekesten_path = Path("~/Documents/datasets/rekesten").expanduser()

    start_of_rekesten = []
    all_rekesten = []
    data = []
    for path in rekesten_path.glob("**/*.jpg"):
        all_rekesten.append(path)
        if path.name in names:
            start_of_rekesten.append(path)
            data.append({"path": path.relative_to(rekesten_path), "name": path.name, "start": 1})
        else:
            data.append({"path": path.relative_to(rekesten_path), "name": path.name, "start": 0})

    df = pd.DataFrame(data)
    df.to_csv("rekesten.csv", index=False)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
