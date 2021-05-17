import pickle, os
import base64
from flask import Flask, request
from src import convert_pdf_to_text

import requests
app = Flask(__name__)
app.config['MAX_CONTENT_PATH'] = '10000000' # 10MB incase of long pathology pdfs

IMGS_DIR = 'tmp/imgs/'
def get_endpoints():
    endpoints = {}
    with open('endpoints.txt','r') as f:
        for line in f.readlines():
            tupl = line.strip().split()
            endpoints[tupl[0]] = tupl[1]
    return endpoints['production'], endpoints['test']


@app.route('/', methods=['POST'])
def run_ocr():
    """
    Expects:
        1 - PDF or PNG of pathology report base64 string in json
        2 - an email address for patient
    :return:
    """
    PRODUCTION_ENDPOINT, TEST_ENDPOINT = get_endpoints()
    print(TEST_ENDPOINT)
    content = request.get_json()
    path_report = base64.b64decode(content['file'])

    input_filename = f'tmp/{content["filename"]}'
    name = os.path.basename(input_filename).split('.')[0]
    with open(input_filename,'wb+') as f:
        f.write(path_report)
        f.close()
    email = content['email']

    if '.pdf' in input_filename:
        file_limit = convert_pdf_to_text._run_pdf(input_filename, IMGS_DIR + name)
        text = convert_pdf_to_text._pages_to_text(file_limit, IMGS_DIR + name, None, write=False)
    else:
        text = convert_pdf_to_text._run_jpg(input_filename, IMGS_DIR)
    m = {'text':text, 'email':email}

    response = requests.post(url=TEST_ENDPOINT, json=m)

    return text


if __name__=='__main__':
    app.run(host='0.0.0.0', port=5002)