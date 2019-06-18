
import argparse
import io
import glob
import os
import re

from skimage import img_as_bool,img_as_uint, img_as_ubyte, io as ioo
from skimage.filters import threshold_otsu
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.morphology import skeletonize, skeletonize_3d, thin, binary_closing
import matplotlib.pyplot as plt

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Default data paths.
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH,
                                  '../labels/2350-common-hangul.txt')
DEFAULT_FONTS_IMAGE_DIR = os.path.join(SCRIPT_PATH, '../image-data/hangul-images-white')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, '../skeleton-image-data')

DEFAULT_LABEL_CSV = os.path.join(SCRIPT_PATH, '../image-data/labels-map.csv')

# Number of random distortion images to generate per font and character.
DISTORTION_COUNT = 3

# Width and height of the resulting image.
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64


def get_binary(img):
    thresh = threshold_otsu(img)
    binary = img > thresh
    return binary


def generate_skeleton_images(labels_csv, label_file, fonts_image_dir, output_dir):
    """Generate Hangul skeleton files.

    This function takes two arguments, i.e. font images whoose skeletons we want to generate
    and output directory where we will store these generated skeleton images and corresponding
    paths. Please make sure that the images are of 64*64 (PNG) size with black backgorund and white 
    character text.
    """

    # Open the labels file from image-data of hangul images
    # 각 이미지의 레이블 정보가 담긴 csv 파일과 모든 레이블이 담긴 txt 파일을 엶
    labels_csv = io.open(labels_csv, 'r', encoding='utf-8')
    labels_file = io.open(label_file, 'r', encoding='utf-8').read().splitlines()

    # Map characters to indices.
    # 레이블 글자(ex: '가')와 인덱스(ex: 1)를 맵핑
    label_dict = {}
    count = 0
    for label in labels_file:
        label_dict[label] = count
        count += 1

    # Build the lists.
    # 각 이미지에 대한 레이블 정보를 인덱스로 변경한 레이블 리스트를 생성
    labels = []
    for row in labels_csv:
        _, label = row.strip().split(',')
        labels.append(label_dict[label])

    # Set the path of skeleton images in output directory. It will be used later for
    # setting up skeleton images path for skeleton labels
    # output 디렉토리 안에 스켈레톤 이미지 경로를 설정 (스케렐톤 이미지가 저장될 경로)
    # 나중에 스켈레톤 레이블을 위한 스켈레톤 이미지 경로를 설정하기 위해 사용될 것임
    image_dir = os.path.join(output_dir, 'skeleton-images-white-otsu-thin')
    if not os.path.exists(image_dir):
        os.makedirs(os.path.join(image_dir))

    # # Set the path of binary images in output directory.
    # binary_dir = os.path.join(output_dir, 'binary-images-white')
    # if not os.path.exists(binary_dir):
    #     os.makedirs(os.path.join(binary_dir))

    # Get the list of all the Hangul images. Sorted function is used to order the arbitrary
    # output of glob() which is always arbitrary
    # 모든 한글 이미지에 대한 리스트를 가져옴
    # 정렬 함수는 항상 임의적인 glob() 임의 출력을 정렬하는데 사용함
    font_images = glob.glob(os.path.join(fonts_image_dir, '*.png'))
    # check if the images are jpeg
    # png 형식으로 읽어들인 이미지가 없다면, 이미지들이 jpeg 형식인지 확인
    if len(font_images) == 0:
        font_images = glob.glob(os.path.join(fonts_image_dir, '*.jpg'))

    # If input directory is empty or no images are found with .png, jpeg and .jpg extension
    # 입력 디렉토리가 비어있거나 png, jpeg, jpg 확장자를 가진 이미지가 없는 경우 예외처리
    if len(font_images) == 0:
        raise Exception("input_dir contains no image files")

    def get_name(path):
        name, _ = os.path.splitext(os.path.basename(path))
        return name

    # If the image names are numbers, sort by the value rather than asciibetically
    # having sorted inputs means that the outputs are sorted in test mode
    # 파일 이름 정렬
    if all(get_name(path).isdigit() for path in font_images):
        font_images = sorted(font_images, key=lambda path: int(get_name(path)))
    else:
        font_images = sorted(font_images)

    #font_images = sorted(font_images, key=lambda x:float(re.findall("(\d+)",x)[0]))

    # Create the skeleton labels file
    # 스켈레톤 레이블 파일(경로와 레이블을 맵핑한 것)을 생성
    labels_csv = io.open(os.path.join(output_dir, 'skeleton-labels-map.txt'), 'w', encoding='utf-8')

    # Set the count so that we can view the progress on the terminal
    # 터미널에서 진전을 보기위해 카운트 설정
    total_count = 0
    prev_count = 0
    index = 0

    # 폰트 이미지들 하나씩 스켈레톤 수행
    for font_image in font_images:
        # Print image count roughly every 5000 images.
        # 매 5000번째 이미지 쯤에서 이미지 카운트 출력
        if total_count - prev_count > 5000:
            prev_count = total_count
            print('{} skeleton images generated...'.format(total_count))

        total_count += 1

        # Read the images one by one from the font images directory
        # Convert them from rgb to gray and then convert it into bool
        # Then apply skeletonize function and wrap it into binary_closing
        # for converting them into binary
        # 폰트 이미지 디렉토리 부터 이미지를 하나씩 읽어들임
        # RGB의 이미지를 그레이스케일로 변형한 뒤, bool 형으로 다시 변형 (이미지 이진화 과정)
        # 골격화 함수를 적용하고 binary_closing 수행(이진 모폴로지 닫기 수행 - 이미지 사이 빈 부분 채우기 위해)

        image = rgb2gray(imread(font_image))

        # Convert gray image to binary
        # gray 이미지를 이진 이미지로 변환
        # image = img_as_bool(image)
        image = get_binary(image)

        # # save binary images
        # # 이진 이미지 저장
        # file_string = '{}.png'.format(total_count)
        # file_path = os.path.join(binary_dir, file_string)
        # binary_image = img_as_uint(image)
        # ioo.imsave(fname=file_path, arr=binary_image)

        # Skeletonize (otsu + thin)
        skeleton = thin(image)
        skeleton = binary_closing(skeleton)

        # convert image as uint before saving in output directory
        # 출력 디렉토리에 저장하기 전에 uint로 이미지를 변환
        skeleton = img_as_uint(skeleton)

        # 이미지 저장
        file_string = '{}.png'.format(total_count)
        file_path = os.path.join(image_dir, file_string)     
        ioo.imsave(fname=file_path, arr=skeleton)

        # Using Labels list and label_dict dictionary we construct the skeleton labels
        # In skeleton label csv file
        # 레이블 리스트와 레이블 딕셔너리를 사용하여 스켈레톤 레이블 작성
        # 스켈레톤 레이블 csv(여기서는 txt 형식)에 저장
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
