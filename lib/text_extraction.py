# This Python file uses the following encoding: utf-8
from lib.words import splitted_words

from google.cloud import vision
from google.cloud.vision import types

from subprocess import call
from utils import rm_dir

import textract
import os

def convert_to_txt(img):
    # Instantiates a client
    client = vision.ImageAnnotatorClient()
    
    with open(img, 'rb') as image_file:
        content = image_file.read()
        
    image = types.Image(content=content)
    # Performs label detection on the image file
    print('Request')
    response = client.text_detection(image=image)
    print('Response')
    labels = response.text_annotations
    try:
        return (labels[0].description)
    except:
        return ""   
    
def images_to_txt_batch(filenames):
    client = vision.client.Client()
    batch = vision.batch.Batch(client)
    images = [types.Image(content=open(file, 'rb').read()) for file in filenames]
    for image in images:
        batch.add_image(image, [vision.feature.Feature('TEXT_DETECTION')])
    results = batch.detect()
    try:
        return '\n'.join(r.full_texts.text for r in results)
    except Exception as e:
        print(e)
        return ''

def batch_list(l, n):
    return [l[i:i+n] for i in range(0,len(l),n)]
    
def images_to_txt(filenames):
    return '\n'.join(images_to_txt_batch(batch) for batch in batch_list(filenames, 16))
    
def extract_text(f, do_ocr=False, tmp_dir='../tmp', min_words=150, pages=5):
    png_dir = tmp_dir + '_' + os.path.basename(f)[:-4]
    try:
        text = textract.process(f)
        if len(splitted_words(text)) <= min_words:
            #print f, u"è una scansione"
            if not do_ocr:
                return ''
            print f, "Estrazione testo con Vision API."
            rm_dir(png_dir)
            os.mkdir(png_dir)
            # out-10.png
            call(("convert -density 300 %s -quality 100 %s" 
                  % (f, os.path.join(png_dir,os.path.basename(f)[:-3]+'png'))).split(' '))
            # os.listdir is not ordered by name, this fixes it
            images = sorted(os.listdir(png_dir), key=lambda item: (int(item.partition('-')[2].partition('.')[0])))
                        
            if len(images) > 0:
                '''
                res = ""
                for i,el in enumerate(images):
                    if i < pages or pages < 0:
                        txt = convert_to_txt(os.path.join(png_dir,el))
                        res += txt
                    else:
                        break
                '''
                images_full = [os.path.join(png_dir,i) for i in images]
                if pages < 0:
                    res = images_to_txt(images_full)
                else:
                    res = images_to_txt(images_full[:pages])
                rm_dir(png_dir)        
                if res != "":
                    rm_dir(png_dir)
                    return res
                else:
                    print f, "is empty"
                    rm_dir(png_dir)
                    return ''
        else:
            rm_dir(png_dir)
            return text
    except Exception as e:
        print e
        #print f, "e' un documento illeggibile..." 
        rm_dir(png_dir)
        return ''