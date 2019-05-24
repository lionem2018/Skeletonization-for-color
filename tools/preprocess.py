# imprts related to creating paths
import io
import os
import argparse

# imports related to preprocess from pix2pix
import tfimage as im
import time
import tensorflow as tf
import numpy as np
import threading


# setting global variable for counter 
index = 0
total_count = 0 

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Default data paths.
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH,
                                  '../labels/2350-common-hangul.txt')
DEFAULT_FONTS_IMAGE_DIR = os.path.join(SCRIPT_PATH, '../image-data/hangul-images')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, '../hangul-skeleton-combine-images')

DEFAULT_LABEL_CSV = os.path.join(SCRIPT_PATH, '../image-data/labels-map.csv')


def combine(src, src_path):
    if args.b_dir is None:
        raise Exception("missing b_dir")

    # find corresponding file in b_dir, could have a different extension
    basename, _ = os.path.splitext(os.path.basename(src_path))
    for ext in [".png", ".jpg"]:
        sibling_path = os.path.join(args.b_dir, basename + ext)
        if os.path.exists(sibling_path):
            sibling = im.load(sibling_path)
            break
    else:
        raise Exception("could not find sibling image for " + src_path)

    # make sure that dimensions are correct
    height, width, _ = src.shape
    if height != sibling.shape[0] or width != sibling.shape[1]:
        raise Exception("differing sizes")
    
    # convert both images to RGB if necessary
    if src.shape[2] == 1:
        src = im.grayscale_to_rgb(images=src)

    if sibling.shape[2] == 1:
        sibling = im.grayscale_to_rgb(images=sibling)

    # remove alpha channel
    if src.shape[2] == 4:
        src = src[:,:,:3]
    
    if sibling.shape[2] == 4:
        sibling = sibling[:,:,:3]

    return np.concatenate([src, sibling], axis=1)

def process(src_path, dst_path, label_dict, labels, labels_csv, image_dir):
    global index
    global total_count

    total_count += 1
    src = im.load(src_path)

    if args.operation == "combine":
        dst = combine(src, src_path)
    else:
        raise Exception("invalid operation")
    im.save(dst, dst_path)

    file_string = '{}.png'.format(total_count)
    file_path = os.path.join(image_dir, file_string)  

    character = list(label_dict.keys())[labels[index]]
    labels_csv.write(u'{},{}\n'.format(file_path, character))
    index += 1 

complete_lock = threading.Lock()
start = None
num_complete = 0
total = 0

def complete():
    global num_complete, rate, last_complete

    with complete_lock:
        num_complete += 1
        now = time.time()
        elapsed = now - start
        rate = num_complete / elapsed
        if rate > 0:
            remaining = (total - num_complete) / rate
        else:
            remaining = 0

        print("%d/%d complete  %0.2f images/sec  %dm%ds elapsed  %dm%ds remaining" % (num_complete, total, rate, elapsed // 60, elapsed % 60, remaining // 60, remaining % 60))

        last_complete = now


def generate_hangul_skeleton_combine_images(labels_csv, label_file, output_dir):
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Open the labels file from image-data of hangul images
    labels_csv = io.open(labels_csv, 'r', encoding='utf-8')
    labels_file = io.open(label_file, 'r',
                      encoding='utf-8').read().splitlines()

    # Map characters to indices.
    label_dict = {}
    count = 0
    for label in labels_file:
        label_dict[label] = count
        count += 1

    # Build the lists.
    labels = []
    for row in labels_csv:
        _, label = row.strip().split(',')
        labels.append(label_dict[label])

    # Set the path of hangul-skeleton-combine images in output directory. It will be used later for 
    # setting up hangul-skeleton-combine images path for hangul-skeleton-combine labels
    image_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(image_dir):
        os.makedirs(os.path.join(image_dir))

    # Create the skeleton labels file
    labels_csv = io.open(os.path.join(args.output_dir, 'skeleton-labels-map.txt'), 'w',
                     encoding='utf-8')

    src_paths = []
    dst_paths = []

    # Check if the directory and images already exsist?
    # If yes then skip those images else create the paths list
    skipped = 0
    for src_path in sorted(im.find(args.input_dir)):
        name, _ = os.path.splitext(os.path.basename(src_path))
        dst_path = os.path.join(image_dir, name + ".png")
        if os.path.exists(dst_path):
            skipped += 1
        else:
            src_paths.append(src_path)
            dst_paths.append(dst_path)
    
    print("skipping %d files that already exist" % skipped)

    global total
    total = len(src_paths)
    
    print("processing %d files" % total)

    global start
    start = time.time()

    if args.workers == 1:
        with tf.Session() as sess:
            for src_path, dst_path in zip(src_paths, dst_paths):
                process(src_path, dst_path, label_dict, labels, labels_csv, image_dir)
                complete()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, 
                            help="path to folder containing images")
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                            default=DEFAULT_OUTPUT_DIR,
                            help='Output directory to store generated hangul skeleton images and '
                                 'label CSV file.')
    parser.add_argument("--operation", required=True, choices=["combine"])
    parser.add_argument("--workers", type=int, default=1, help="number of workers")
    # combine
    parser.add_argument("--b_dir", type=str, help="path to folder containing B images for combine operation")
    # labels
    parser.add_argument('--image-label-csv', type=str, dest='labels_csv',
                        default=DEFAULT_LABEL_CSV,
                        help='File containing image paths and corresponding '
                             'labels.')
    parser.add_argument('--label-file', type=str, dest='label_file',
                        default=DEFAULT_LABEL_FILE,
                        help='File containing newline delimited labels.')
    args = parser.parse_args()

    generate_hangul_skeleton_combine_images(args.labels_csv, args.label_file, args.output_dir)