# Import libraries 
from PIL import Image 
import pytesseract 
import sys 
from pdf2image import convert_from_path
import numpy as np
import cv2
import os 
import glob
import tempfile



IMAGE_SIZE = 1800
BINARY_THREHOLD = 180


def process_image_for_ocr(file_path):
    # TODO : Implement using opencv
    temp_filename = set_image_dpi(file_path)
    im_new = remove_noise_and_smooth(temp_filename)
    return im_new


def set_image_dpi(file_path):
    im = Image.open(file_path)
    length_x, width_y = im.size
    factor = max(1, int(IMAGE_SIZE / length_x))
    size = factor * length_x, factor * width_y
    # size = (1800, 1800)
    im_resized = im.resize(size, Image.ANTIALIAS)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_filename = temp_file.name
    im_resized.save(temp_filename, dpi=(400, 400))
    return temp_filename


def image_smoothening(img):
    ret1, th1 = cv2.threshold(img, BINARY_THREHOLD, 255, cv2.THRESH_BINARY)
    ret2, th2 = cv2.threshold(th1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(th2, (1, 1), 0)
    ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3


def remove_noise_and_smooth(file_name):
    img = cv2.imread(file_name, 0)
    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41,
                                     3)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    img = image_smoothening(img)
    or_image = cv2.bitwise_or(img, closing)
    return or_image


def _pages_to_text(filelimit, out_dir, output_file, write=True):
    text_arr = []
    for i in range(1, filelimit + 1):
        # Set filename to recognize text from
        # Again, these files will be:
        # page_1.jpg
        # page_2.jpg
        # ....
        # page_n.jpg
        filename = "{}\\page_{}.jpg".format(out_dir, str(i))

        img = process_image_for_ocr(filename)
        text = pytesseract.image_to_string(img, config="-l eng --oem 3 --psm 6")

        text = text.replace('-\n', '')
        text_arr.append(text)
        # Finally, write the processed text to the file.
        if write:
            output_file.write(text)
    return '\n'.join(text_arr)

def _run_pdf(PDF_file, out_dir):
    print("Processing " + PDF_file + "...")
    ''' 
    Part #1 : Converting PDF to images 
    '''

    # Store all the pages of the PDF in a variable
    pages = convert_from_path(PDF_file, 500)

    # Counter to store images of each page of PDF to image
    image_counter = 1

    # Iterate through all the pages stored above
    for page in pages:
        # Declaring filename for each page of PDF as JPG
        # For each page, filename will be:
        # PDF page 1 -> page_1.jpg
        # PDF page 2 -> page_2.jpg
        # PDF page 3 -> page_3.jpg
        # ....
        # PDF page n -> page_n.jpg
        filename = "{}\\page_{}.jpg".format(out_dir, str(image_counter))

        # Save the image of the page in system
        page.save(filename, 'JPEG')
        print(f'saved:'
              f't{filename}')
        # Increment the counter to update filename
        image_counter +=1

    return image_counter - 1

def run_pdf(data_dir, out_dir):
    print("OCR Conversion starting...")
    all_pdfs = glob.glob(data_dir + '/*.pdf')
    for PDF_file in all_pdfs:
        print("Processing "+PDF_file+"...")
        ''' 
        Part #1 : Converting PDF to images 
        '''
        # Counter to store images of each page of PDF to image
        image_counter = _run_pdf(PDF_file,out_dir)

        ''' 
        Part #2 - Recognizing text from the images using OCR 
        '''

        # Variable to get count of total number of pages
        filelimit = image_counter-1
        # Creating a text file to write the output
        basename = os.path.basename(PDF_file)
        outfile = "{}\\{}.txt".format(out_dir, basename.replace(".pdf",""))

        # Open the file in append mode so that
        # All contents of all images are added to the same file
        f = open(outfile, 'w+', encoding='utf-8')

        fulltext = _pages_to_text(filelimit, out_dir,f)

        # Close the file after writing all the text.
        f.close()


def _run_jpg(JPG_file, out_dir,write=True):
    print("Processing " + JPG_file + "...")

    basename = os.path.basename(JPG_file)
    outfile = "{}\\{}.txt".format(out_dir, basename.replace(".JPEG", ""))

    filename = JPG_file

    img = process_image_for_ocr(filename)
    text = pytesseract.image_to_string(img, config="-l eng --oem 3 --psm 6")

    text = text.replace('-\n', '')

    # Finally, write the processed text to the file.
    if write:
        f = open(outfile, 'w+', encoding='utf-8')
        f.write(text)

    # Close the file after writing all the text.
    f.close()
    return text

def run_jpg(data_dir, out_dir):
    print("OCR Conversion starting...")
    all_jpgs = glob.glob(data_dir + '/*.JPEG')
    for JPG_file in all_jpgs:
        text = _run_jpg(JPG_file,out_dir)
    return None


def clean_up(out_dir):
    filelist = glob.glob(os.path.join(out_dir, "*.jpg"))
    for f in filelist:
        os.remove(f)


if __name__ == '__main__':
    if sys.argv[1] == 'help':
        print(
            "@ 1 - Directory containing text data output\n"
            "@ 2 - Directory containing corresponding eval data\n"
        )

    run_jpg(sys.argv[1], sys.argv[2])
    run_pdf(sys.argv[1], sys.argv[2])
    clean_up(sys.argv[2])
    print("Processing Complete")

