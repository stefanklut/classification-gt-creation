import argparse

# from multiprocessing.pool import ThreadPool as Pool
import os
import sys
from collections import Counter
from multiprocessing.pool import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from page_xml.xmlPAGE import PageData
from utils.input_utils import get_file_paths, supported_image_formats
from utils.path_utils import image_path_to_xml_path


def pretty_print(input_dict: dict[str, float], n_decimals=3):
    """
    Print the dict with better readability

    Args:
        input_dict (dict[str, float]): dictionary of
        n_decimals (int, optional): rounding of the float values. Defaults to 3.
    """
    len_names = max(len(str(key)) for key in input_dict.keys()) + 1
    len_values = max(len(f"{value:.{n_decimals}f}") for value in input_dict.values()) + 1

    output_string = ""
    for key, value in input_dict.items():
        output_string += f"{str(key):<{len_names}}: {value:<{len_values}.{n_decimals}f}\n"

    print(output_string)


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count regions from a dataset")

    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-i", "--input", help="Input folder/file", nargs="+", action="extend", type=str)
    io_args.add_argument("-c", "--classes", help="Path to the classes file", type=str, required=True)
    args = parser.parse_args()
    return args


def count_regions_single_page(xml_path: Path) -> tuple[Counter, Path]:
    """
    Count the unique regions in a pageXML

    Args:
        xml_path (Path): Path to pageXML

    Returns:
        Counter: Count of all unique regions
    """
    page_data = PageData(xml_path)
    page_data.parse()

    region_names = ["TextRegion"]  # Assuming this is all there is
    zones = page_data.get_zones(region_names)

    if zones is None:
        return Counter(), xml_path

    counter = Counter(item["type"] for item in zones.values())
    return counter, xml_path


def main(args):
    """
    Run the full count over all pageXMLs found in the input dir

    Args:
        args (argparse.Namespace): command line arguments
    """

    image_paths = get_file_paths(args.input, supported_image_formats)
    xml_paths = [image_path_to_xml_path(image_path) for image_path in image_paths]
    data = pd.read_excel(args.classes)

    classes = {}
    for index, row in data.iterrows():
        filename = row["Bestandsnamen"]
        if isinstance(filename, float) and np.isnan(filename):
            print(f"NaN found at index {index}")
            continue
        elif not isinstance(filename, str):
            print(f"Unknown type found at index {index}")
            continue
        name = Path(filename).stem
        classes[name] = 1

    # xml_paths = get_file_paths(args.input, [".xml"])

    # Single thread
    # regions_per_page = []
    # for xml_path_i in tqdm(xml_paths):
    #     regions_per_page.extend(count_regions_single_page(xml_path_i))

    # Multithread
    with Pool(os.cpu_count()) as pool:
        regions_per_page = list(
            tqdm(
                iterable=pool.imap_unordered(count_regions_single_page, xml_paths),
                total=len(xml_paths),
                desc="Extracting Regions",
            )
        )

    # Combine the counters of multiple regions
    start_counters = [counter for counter, xml_path in regions_per_page if classes.get(xml_path.stem, 0) == 1]
    not_start_counters = [counter for counter, xml_path in regions_per_page if classes.get(xml_path.stem, 0) == 0]

    print("Start")
    total_start_regions = sum(start_counters, Counter())
    print(f"Total: {len(start_counters)}")
    pretty_print(dict(total_start_regions), n_decimals=0)

    keys = list(total_start_regions.keys())
    contains_region = {key: 0 for key in keys}
    for key in keys:
        for region in start_counters:
            if key in region:
                contains_region[key] += 1
    percentage = {key: value / len(start_counters) for key, value in contains_region.items()}
    print("Percentage")
    pretty_print(percentage)

    print("Average")
    average = {key: value / len(start_counters) for key, value in total_start_regions.items()}
    pretty_print(average)

    print("Not start")
    total_not_start_regions = sum(not_start_counters, Counter())
    print(f"Total: {len(not_start_counters)}")
    pretty_print(dict(total_not_start_regions), n_decimals=0)

    keys = list(total_not_start_regions.keys())
    contains_region = {key: 0 for key in keys}
    for key in keys:
        for region in not_start_counters:
            if key in region:
                contains_region[key] += 1
    percentage = {key: value / len(not_start_counters) for key, value in contains_region.items()}
    print("Percentage")
    pretty_print(percentage)

    print("Average")
    average = {key: value / len(not_start_counters) for key, value in total_not_start_regions.items()}
    pretty_print(average)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
