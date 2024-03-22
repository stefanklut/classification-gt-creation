import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.joinpath("..")))
from page_xml.xmlPAGE import PageData
from utils.copy_utils import copy_mode
from utils.input_utils import get_file_paths, supported_image_formats
from utils.path_utils import image_path_to_xml_path, xml_path_to_image_path


def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Count regions from a dataset")

    io_args = parser.add_argument_group("IO")
    io_args.add_argument("-i", "--input", help="Input folder/file", nargs="+", action="extend", type=str)
    io_args.add_argument("-c", "--classes", help="Path to the classes file", type=str, required=True)

    parser.add_argument("-o", "--output", help="Output folder", type=str)

    parser.add_argument("--copy", help="Copy the files to the output folder", action="store_true")
    parser.add_argument("--confusion-matrix", help="Save the confusion matrix to a file", action="store_true")
    parser.add_argument("--dates", help="Save the dates to a file", action="store_true")

    args = parser.parse_args()
    return args


def predict_class(xml_path: Path) -> tuple[int, dict]:
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
        return 0, {}
    found_metadata_cover = False
    found_date = False
    found_petitioner = False
    found_signature_mark = False
    found_addressing = False
    date_text = {}

    for item in zones.values():
        if item["type"] == "metadata\\u0020cover":
            found_metadata_cover = True
        elif item["type"] == "date":
            for textline in item["node"].findall("".join(["./", page_data.base, "TextLine"])):
                if not textline:
                    print(textline)
                    continue
                text = str(page_data.get_text(textline))
                date_text[item["id"]] = text

            found_date = True
        elif item["type"] == "petitioner":
            found_petitioner = True
        elif item["type"] == "signature-mark":
            found_signature_mark = True
        elif item["type"] == "addressing":
            found_addressing = True

    # if found_metadata_cover:
    #     return 1, date_text

    # if found_metadata_cover and found_date and found_petitioner:
    #     return 1, date_text

    # if found_metadata_cover and found_date and found_signature_mark:
    #     return 1, date_text

    # if found_date and found_petitioner and found_signature_mark:
    #     return 1, date_text

    # if found_addressing:
    #     return 1, date_text

    if found_metadata_cover and found_date and found_petitioner and found_signature_mark:
        return 1, date_text

    return 0, date_text
    # counter = Counter(item["type"] for item in zones.values())

    # return counter


def confusion_matrix(gt: np.ndarray, pred: np.ndarray, n_classes: Optional[int] = None) -> np.ndarray:
    """
    Create a confusion matrix from ground truth and predictions

    Args:
        gt (np.ndarray): Ground truth
        pred (np.ndarray): Predictions

    Returns:
        np.ndarray: Confusion matrix
    """
    assert len(gt) == len(pred), "Ground truth and predictions must be of the same length"

    if n_classes is None:
        n_classes = len(np.unique(gt))
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)

    for i in range(len(gt)):
        confusion_matrix[gt[i], pred[i]] += 1

    return confusion_matrix


def confusion_matrix_xmls(gt: np.ndarray, pred: np.ndarray, xml_paths: Sequence[Path], n_classes: Optional[int] = None) -> dict:
    assert len(gt) == len(pred), "Ground truth and predictions must be of the same length"

    if n_classes is None:
        n_classes = len(np.unique(gt))

    confusion_matrix_xmls = {i: {j: [] for j in range(n_classes)} for i in range(n_classes)}

    for i in range(len(gt)):
        confusion_matrix_xmls[gt[i]][pred[i]].append(str(xml_paths[i]))

    return confusion_matrix_xmls


def precision(confusion_matrix: np.ndarray):
    """
    Calculate the precision from a confusion matrix

    Args:
        confusion_matrix (np.ndarray): Confusion matrix

    Returns:
        np.ndarray: Precision
    """
    return np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=0)


def recall(confusion_matrix: np.ndarray):
    """
    Calculate the recall from a confusion matrix

    Args:
        confusion_matrix (np.ndarray): Confusion matrix

    Returns:
        np.ndarray: Recall
    """
    return np.diag(confusion_matrix) / np.sum(confusion_matrix, axis=1)


def accuracy(confusion_matrix: np.ndarray):
    """
    Calculate the accuracy from a confusion matrix

    Args:
        confusion_matrix (np.ndarray): Confusion matrix

    Returns:
        np.ndarray: Accuracy
    """
    return np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)


def main(args):
    """
    Run the full count over all pageXMLs found in the input dir

    Args:
        args (argparse.Namespace): command line arguments
    """
    image_paths = get_file_paths(args.input, supported_image_formats)
    xml_paths = [image_path_to_xml_path(image_path, check=False) for image_path in image_paths]

    existing_xml_paths = []
    for xml_path in xml_paths:
        if xml_path.exists():
            existing_xml_paths.append(xml_path)
        else:
            # print(f"PageXML not found for {xml_path}")
            pass

    xml_paths = existing_xml_paths

    # Single thread
    # regions_per_page = []
    # for xml_path_i in tqdm(xml_paths):
    #     regions_per_page.extend(count_regions_single_page(xml_path_i))

    # Multithread
    # with Pool(os.cpu_count()) as pool:
    #     regions_per_page = list(
    #         tqdm(
    #             iterable=pool.imap_unordered(count_regions_single_page, xml_paths),
    #             total=len(xml_paths),
    #             desc="Extracting Regions",
    #         )
    #     )

    # Combine the counters of multiple regions

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

    gt = np.array([classes.get(xml_path.stem, 0) for xml_path in xml_paths])
    total_date_dict = {}
    pred = []
    for xml_path in xml_paths:
        pred_i, date_dict = predict_class(xml_path)
        pred.append(pred_i)

        if date_dict:
            path_key = str(xml_path)
            if path_key in total_date_dict:
                total_date_dict[path_key].update(date_dict)
            else:
                total_date_dict[path_key] = date_dict
    pred = np.array(pred)

    cm = confusion_matrix(gt, pred, n_classes=2)
    cm_xmls = confusion_matrix_xmls(gt, pred, xml_paths)

    if args.output:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)

        if args.dates:
            with open(output.joinpath("dates.json"), "w") as f:
                json.dump(total_date_dict, f)
        if args.confusion_matrix:
            with open(output.joinpath("confusion_matrix.json"), "w") as f:
                json.dump(cm_xmls, f)

        if args.copy:
            for cm_class, cm_dict in cm_xmls.items():
                for cm_pred, xml_files in cm_dict.items():
                    for xml_file in xml_files:
                        xml_file = Path(xml_file)
                        image_file = xml_path_to_image_path(xml_file)
                        image_dir = output.joinpath(f"{cm_class}_{cm_pred}")
                        page_dir = image_dir.joinpath("page")
                        image_dir.mkdir(parents=True, exist_ok=True)
                        page_dir.mkdir(parents=True, exist_ok=True)
                        copy_mode(xml_file, page_dir.joinpath(xml_file.name), mode="link")
                        copy_mode(image_file, image_dir.joinpath(image_file.name), mode="link")

    prec = precision(cm)
    rec = recall(cm)
    acc = accuracy(cm)

    print("Confusion matrix")
    print(cm)
    print("Precision")
    print(prec)
    print("Recall")
    print(rec)
    print("Accuracy")
    print(acc)


if __name__ == "__main__":
    args = get_arguments()
    main(args)
