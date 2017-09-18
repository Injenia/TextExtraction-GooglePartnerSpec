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
    response = client.text_detection(image=image)
    labels = response.text_annotations
    try:
        return (labels[0].description)
    except:
        return ""   

def extract_text(f, do_ocr, png_dir='../tmp', min_words=150, pages=5):
    try:
        text = textract.process(f)
        if len(splitted_words(text)) <= min_words:
            #print f, u"è una scansione"
            if not do_ocr:
                return ''
            print "Estrazione testo con Vision API."
            rm_dir(png_dir)
            os.mkdir(png_dir)
            # out-10.png
            call(("convert -verbose -density 300 %s -quality 100 %s" 
                  % (f, os.path.join(png_dir,os.path.basename(f)[:-3]+'png'))).split(' '))
            # os.listdir is not ordered by name, this fixes it
            images = sorted(os.listdir(png_dir), key=lambda item: (int(item.partition('-')[2].partition('.')[0])))
                        
            if len(images) > 0:
                res = ""
                for i,el in enumerate(images):
                    if i < pages:
                        txt = convert_to_txt(os.path.join(png_dir,el))
                        res += txt
                    else:
                        break
                rm_dir(png_dir)        
                if res != "":
                    return res
                else:
                    print f, "is empty"
                    return ''
        else:
            return text
    except Exception as e:
        print e
        #print f, "e' un documento illeggibile..." 
        rm_dir(png_dir)
        return ''