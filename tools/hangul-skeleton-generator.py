
import argparse
import io
import glob
import os
import re

from skimage import img_as_bool,img_as_uint, io as ioo
from skimage.filters import threshold_otsu
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.morphology import skeletonize, binary_closing, thin

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Default data paths.
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH,
                                  '../labels/2350-common-hangul.txt')
DEFAULT_FONTS_IMAGE_DIR = os.path.join(SCRIPT_PATH, '../image-data/hangul-images')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, '../skeleton-image-data')

DEFAULT_LABEL_CSV = os.path.join(SCRIPT_PATH, '../image-data/labels-map.csv')

# Number of random distortion images to generate per font and character.
DISTORTION_COUNT = 3

# Width and height of the resulting image.
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

# def get_binary(img):    
#     thresh = threshold_otsu(img)
#     binary = img > thresh
#     return binary

def generate_skeleton_images(labels_csv, label_file, fonts_image_dir, output_dir):
    """Generate Hangul skeleton files.

    This function takes two arguments, i.e. font images whoose skeletons we want to generate
    and output directory where we will store these generated skeleton images and corresponding
    paths. Please make sure that the images are of 64*64 (PNG) size with black backgorund and white 
    character text.
    """

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

    # Set the path of skeleton images in output directory. It will be used later for 
    # setting up skeleton images path for skeleton labels
    image_dir = os.path.join(output_dir, 'skeleton-images')
    if not os.path.exists(image_dir):
        os.makedirs(os.path.join(image_dir))

    # Get the list of all the Hangul images. Sorted function is used to order the arbitray 
    # output of glob() which is always arbitray
    font_images = glob.glob(os.path.join(fonts_image_dir, '*.png'))
    # check if the images are jpeg
    if len(font_images) == 0:
        font_images = glob.glob(os.path.join(fonts_image_dir, '*.jpg'))

    # If input directory is empty or no images are found with .png, jpeg and .jpg extension
    if len(font_images) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # If the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    if all(get_name(path).isdigit() for path in font_images):
        font_images = sorted(font_images, key=lambda path: int(get_name(path)))
    else:
        font_images = sorted(font_images)

    #font_images = sorted(font_images, key=lambda x:float(re.findall("(\d+)",x)[0]))

    # Create the skeleton labels file
    labels_csv = io.open(os.path.join(output_dir, 'skeleton-labels-map.txt'), 'w',
                        encoding='utf-8')

    # Set the count so that we can view the progress on the terminal
    total_count = 0
    prev_count= 0
    index = 0

    for font_image in font_images:
        # Print image count roughly every 5000 images.
        if total_count - prev_count > 5000:
            prev_count = total_count
            print('{} skeleton images generated...'.format(total_count))

        total_count += 1

        # Read the images one by one from the font images directory
        # Convert them from rgb to gray and then convert it into bool
        # Then apply skeletonize function and wrap it into binary_closing
        # for converting them into binary

        image = img_as_bool(rgb2gray(imread(font_image)))
        skeleton = binary_closing(thin(image))

        # convert image as uint before saving in output directory
        skeleton = img_as_uint(skeleton)

        file_string = '{}.png'.format(total_count)
        file_path = os.path.join(image_dir, file_string)     
        ioo.imsave(fname=file_path, arr=skeleton)

        # Using Labels list and label_dict dictionary we construct the skeleton labels
        # In skeleton label csv file
        character = list(label_dict.keys())[labels[index]]
        labels_csv.write(u'{},{}\n'.format(file_path, character))
        index += 1    



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image-label-csv', type=str, dest='labels_csv',
                        default=DEFAULT_LABEL_CSV,
                        help='File containing image paths and corresponding '
                             'labels.')
    parser.add_argument('--label-file', type=str, dest='label_file',
                        default=DEFAULT_LABEL_FILE,
                        help='File containing newline delimited labels.')
    parser.add_argument('--font-image-dir', type=str, dest='fonts_image_dir',
                        default=DEFAULT_FONTS_IMAGE_DIR,
                        help='Directory of images to use for extracting skeletons.')
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR,
                        help='Output directory to store generated skeleton images and '
                             'label CSV file.')
    args = parser.parse_args()
    generate_skeleton_images(args.labels_csv, args.label_file, args.fonts_image_dir, args.output_dir)