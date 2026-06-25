#!/usr/bin/env python3
import argparse
from pathlib import Path

import cv2
import numpy as np


APRILTAG_DICTIONARIES = {
    "16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "36h11": cv2.aruco.DICT_APRILTAG_36h11,
}


def make_labeled_tag(marker, marker_id, padding):
    marker = cv2.copyMakeBorder(
        marker,
        padding,
        padding * 2,
        padding,
        padding,
        cv2.BORDER_CONSTANT,
        value=255,
    )
    label = f"AprilTag {marker_id}"
    cv2.putText(
        marker,
        label,
        (padding, marker.shape[0] - padding // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        0,
        2,
        cv2.LINE_AA,
    )
    return marker


def generate_tags(ids, output_dir, dictionary_name, size, padding, labeled):
    output_dir.mkdir(parents=True, exist_ok=True)
    dictionary = cv2.aruco.getPredefinedDictionary(APRILTAG_DICTIONARIES[dictionary_name])

    paths = []
    for marker_id in ids:
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, size)
        image = make_labeled_tag(marker, marker_id, padding) if labeled else marker
        path = output_dir / f"apriltag_{dictionary_name}_{marker_id}.png"
        cv2.imwrite(str(path), image)
        paths.append(path)
    return paths


def make_contact_sheet(paths, output_dir, dictionary_name):
    images = [cv2.imread(str(path), cv2.IMREAD_GRAYSCALE) for path in paths]
    if not images:
        return None

    height = max(image.shape[0] for image in images)
    normalized = []
    for image in images:
        if image.shape[0] != height:
            scale = height / image.shape[0]
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        normalized.append(image)

    spacer = np.full((height, 40), 255, dtype=np.uint8)
    sheet = normalized[0]
    for image in normalized[1:]:
        sheet = np.hstack([sheet, spacer, image])

    path = output_dir / f"apriltag_{dictionary_name}_sheet.png"
    cv2.imwrite(str(path), sheet)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Generate AprilTag PNG images.")
    parser.add_argument("--ids", nargs="+", type=int, default=[0, 1, 2, 3], help="Tag IDs to generate.")
    parser.add_argument("--out", default="apriltags", help="Output directory.")
    parser.add_argument("--dict", choices=sorted(APRILTAG_DICTIONARIES), default="36h11", help="AprilTag dictionary.")
    parser.add_argument("--size", type=int, default=600, help="Marker image size in pixels.")
    parser.add_argument("--padding", type=int, default=60, help="White padding around labeled tags.")
    parser.add_argument("--no-label", action="store_true", help="Do not add text labels.")
    parser.add_argument("--sheet", action="store_true", help="Also create one combined contact sheet.")
    return parser.parse_args()


def main():
    args = parse_args()
    paths = generate_tags(
        ids=args.ids,
        output_dir=Path(args.out),
        dictionary_name=args.dict,
        size=args.size,
        padding=args.padding,
        labeled=not args.no_label,
    )
    sheet = make_contact_sheet(paths, Path(args.out), args.dict) if args.sheet else None

    for path in paths:
        print(path)
    if sheet is not None:
        print(sheet)


if __name__ == "__main__":
    main()
