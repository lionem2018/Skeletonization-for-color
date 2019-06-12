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
# 카운터를 위한 글로벌 변수 설정
index = 0
total_count = 0 

SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

# Default data paths.
# 디폴드 데이터 경로
DEFAULT_LABEL_FILE = os.path.join(SCRIPT_PATH, '../labels/2350-common-hangul.txt')
DEFAULT_FONTS_IMAGE_DIR = os.path.join(SCRIPT_PATH, '../image-data/hangul-images-white')
DEFAULT_ORIGINAL_SKELETON_IMAGE_DIR = os.path.join(SCRIPT_PATH, '../skeleton-image-data/skeleton-images-white')
DEFAULT_OTSU_SKELETON_IMAGE_DIR = os.path.join(SCRIPT_PATH, '../skeleton-image-data/skeleton-images-white-otsu')
DEFAULT_MAX_ITER_SKELETON_IMAGE_DIR = os.path.join(SCRIPT_PATH, '../skeleton-image-data/skeleton-images-white-otsu-max-iter')
DEFAULT_SKELETON_SKELETON_IMAGE_DIR = os.path.join(SCRIPT_PATH, '../skeleton-image-data/skeleton-images-white-otsu-skeleton')
DEFAULT_SKELETON3D_SKELETON_IMAGE_DIR = os.path.join(SCRIPT_PATH, '../skeleton-image-data/skeleton-images-white-otsu-3d')
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_PATH, '../hangul-skeleton-combine-images')
DEFAULT_LABEL_CSV = os.path.join(SCRIPT_PATH, '../image-data/labels-map.csv')


# 하나의 한글 이미지 정보를 입력 받아 스켈레톤 이미지를 찾아 전처리한 후 나란히 연결함
def combine(src, src_path):
    if args.b1_dir is None:
        raise Exception("missing b1_dir")
    elif args.b2_dir is None:
        raise Exception("missing b2_dir")
    elif args.b3_dir is None:
        raise Exception("missing b3_dir")
    elif args.b4_dir is None:
        raise Exception("missing b4_dir")
    elif args.b5_dir is None:
        raise Exception("missing b5_dir")

    # find corresponding file in b_dir, could have a different extension
    # b_dir에서 해당 파일 탐색, 다른 확장자를 가질 수도 있음
    # 입력받은 한글 이미지(src_path)의 확장자를 뺀 이름만 가져와
    # 해당 이름을 가진 스켈레톤 이미지의 경로를 얻음(여기서는 sibling-형제-으로 표현)
    basename, _ = os.path.splitext(os.path.basename(src_path))
    for ext in [".png", ".jpg"]:
        sibling_path1 = os.path.join(args.b1_dir, basename + ext)
        if os.path.exists(sibling_path1):
            sibling1 = im.load(sibling_path1)
            break
    else:
        raise Exception("could not find sibling1 image for " + src_path)

    for ext in [".png", ".jpg"]:
        sibling_path2 = os.path.join(args.b2_dir, basename + ext)
        if os.path.exists(sibling_path2):
            sibling2 = im.load(sibling_path2)
            break
    else:
        raise Exception("could not find sibling2 image for " + src_path)

    for ext in [".png", ".jpg"]:
        sibling_path3 = os.path.join(args.b3_dir, basename + ext)
        if os.path.exists(sibling_path3):
            sibling3 = im.load(sibling_path3)
            break
    else:
        raise Exception("could not find sibling3 image for " + src_path)

    for ext in [".png", ".jpg"]:
        sibling_path4 = os.path.join(args.b4_dir, basename + ext)
        if os.path.exists(sibling_path4):
            sibling4 = im.load(sibling_path4)
            break
    else:
        raise Exception("could not find sibling2 image for " + src_path)

    for ext in [".png", ".jpg"]:
        sibling_path5 = os.path.join(args.b5_dir, basename + ext)
        if os.path.exists(sibling_path5):
            sibling5 = im.load(sibling_path5)
            break
    else:
        raise Exception("could not find sibling2 image for " + src_path)

    # make sure that dimensions are correct
    # 한글 이미지의 크기와 스켈레톤 이미지의 크기가 같은지 확인
    height, width, _ = src.shape
    if height != sibling1.shape[0] or width != sibling1.shape[1] or height != sibling2.shape[0] or width != sibling2.shape[1]:
        raise Exception("differing sizes")
    
    # convert all images to RGB if necessary
    # 두 이미지가 만일 그레이스케일 이미지라면 RGB로 변환
    if src.shape[2] == 1:
        src = im.grayscale_to_rgb(images=src)

    if sibling1.shape[2] == 1:
        sibling1 = im.grayscale_to_rgb(images=sibling1)

    if sibling2.shape[2] == 1:
        sibling2 = im.grayscale_to_rgb(images=sibling2)

    if sibling3.shape[2] == 1:
        sibling3 = im.grayscale_to_rgb(images=sibling3)

    if sibling4.shape[2] == 1:
        sibling4 = im.grayscale_to_rgb(images=sibling4)

    if sibling5.shape[2] == 1:
        sibling5 = im.grayscale_to_rgb(images=sibling5)

    # remove alpha channel
    # 두 이미지의 알파 채널(투명도)를 삭제함
    if src.shape[2] == 4:
        src = src[:,:,:3]
    
    if sibling1.shape[2] == 4:
        sibling1 = sibling1[:,:,:3]

    if sibling2.shape[2] == 4:
        sibling2 = sibling2[:,:,:3]

    if sibling3.shape[2] == 4:
        sibling3 = sibling3[:,:,:3]

    if sibling4.shape[2] == 4:
        sibling4 = sibling4[:,:,:3]

    if sibling5.shape[2] == 4:
        sibling5 = sibling5[:,:,:3]


    # 두 이미지를 하나로 합쳐 리턴
    return np.concatenate([src, sibling1, sibling2, sibling3, sibling4, sibling5], axis=1)


def process(src_path, dst_path, label_dict, labels, labels_csv, image_dir):
    global index
    global total_count

    # 입력받은 하나의 한글 이미지 경로를 가지고 해당 한글 이미지를 읽음
    total_count += 1
    src = im.load(src_path)

    # 명령어에서 인자로 입력받은 연산이 combine이라면 combine 수행
    if args.operation == "combine":
        dst = combine(src, src_path)
    else:
        raise Exception("invalid operation")
    # combine을 수행한 결과를 파일로 저장함
    im.save(dst, dst_path)

    # 저장한 combine 결과 이미지를 csv 파일에 레이블을 맵핑하여 저장
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

        print("%d/%d complete  %0.2f images/sec  %dm%ds elapsed  %dm%ds remaining"
              % (num_complete, total, rate, elapsed // 60, elapsed % 60, remaining // 60, remaining % 60))

        last_complete = now


def generate_hangul_skeleton_combine_images(labels_csv, label_file, output_dir):
    # 출력 디렉토리 유무 확인 후 생성
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Open the labels file from image-data of hangul images
    # 한글 이미지가 있는 image-data 폴더로부터 레이블 파일을 읽어들임
    labels_csv = io.open(labels_csv, 'r', encoding='utf-8')
    labels_file = io.open(label_file, 'r', encoding='utf-8').read().splitlines()

    # Map characters to indices.
    # 글자 레이블 정보와 인덱스를 맵핑
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

    # Set the path of hangul-skeleton-combine images in output directory. It will be used later for 
    # setting up hangul-skeleton-combine images path for hangul-skeleton-combine labels
    # output 디렉토리 안에 combine 결과 이미지 경로를 설정 (이미지가 저장될 경로)
    # 나중에 combine 레이블을 위한 combine 결과 이미지 경로를 설정하기 위해 사용될 것임
    image_dir = os.path.join(output_dir, 'images-white-all')
    if not os.path.exists(image_dir):
        os.makedirs(os.path.join(image_dir))

    # Create the skeleton labels file
    # combine 결과 이미지 레이블 파일 생성
    labels_csv = io.open(os.path.join(args.output_dir, 'skeleton-labels-map.txt'), 'w', encoding='utf-8')

    src_paths = []
    dst_paths = []

    # Check if the directory and images already exist?
    # If yes then skip those images else create the paths list
    # combine 결과 이미지가 이미 존재하는 지 확인
    # 만약 존재한다면 그 이미지들을 스킵하고, 그렇지 않으면 경로 리스트를 생성함
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
    parser.add_argument("--input_dir",
                            default=DEFAULT_FONTS_IMAGE_DIR,
                            help="path to folder containing images")
    parser.add_argument('--output-dir', type=str, dest='output_dir',
                            default=DEFAULT_OUTPUT_DIR,
                            help='Output directory to store generated hangul skeleton images and '
                                 'label CSV file.')
    parser.add_argument("--operation", default='combine', choices=["combine"])
    parser.add_argument("--workers", type=int, default=1, help="number of workers")
    # combine
    parser.add_argument("--b1_dir", type=str, default=DEFAULT_ORIGINAL_SKELETON_IMAGE_DIR,
                        help="path to folder containing B images of white characters for combine operation")
    parser.add_argument("--b2_dir", type=str, default=DEFAULT_OTSU_SKELETON_IMAGE_DIR,
                        help="path to folder containing B images of white characters for combine operation")
    parser.add_argument("--b3_dir", type=str, default=DEFAULT_MAX_ITER_SKELETON_IMAGE_DIR,
                        help="path to folder containing B images of white characters for combine operation")
    parser.add_argument("--b4_dir", type=str, default=DEFAULT_SKELETON_SKELETON_IMAGE_DIR,
                        help="path to folder containing B images of white characters for combine operation")
    parser.add_argument("--b5_dir", type=str, default=DEFAULT_SKELETON3D_SKELETON_IMAGE_DIR,
                        help="path to folder containing B images of white characters for combine operation")
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
