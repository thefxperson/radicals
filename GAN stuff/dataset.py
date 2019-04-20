import tensorflow as tf
import struct
import numpy as np
import random
from PIL import Image, ImageEnhance


#retrieve a record from the designated file. returns a tuple of a numpy array of the image, and the kanji encoded in utf-8.
def retrieve(filename, record_id):
    #open file and find corresponding record. then read record and close file
    #each record is of size 8199 bytes
    file = open(filename, "rb")
    file.seek(record_id * 8199)
    s = file.read(8199)
    file.close()

    #unpack binary data from file into data array
    #data is stored as followed:
    #0 - Int - Serial Sheet Number
    #1 - Int - JIS Character Code (JIS X 0208)
    #2 - Bytes - JIS Typical Reading (string)
    #3 - Int - Serial Data Number
    #4-9 - Metadata. Likely all 0s.
    #10 - Date of Collection (19YYMM)
    #11 - Date of Scan (19YYMM)
    #12 - X-Coordinate of Sample on Sheet
    #13 - Y-Coordinate of Sample on Sheet
    #14 - Bytes - Image (4bit 128x127 image)
    #For more detailed information see ETL-9G: http://etlcdb.db.aist.go.jp/?page_id=1711 
    data = struct.unpack(">2H8sI4B4H2B34x8128s7x", s)

    #convert from bytes to a properly encoded string
    #it is unclear if character is encoded using the 1978 or 1983 JIS X 0208 standard. 83 is used here.
    #b"\033$B" is the ISO2022 escape code from JIS X 0208-1983. b"\033$@" is for JIX X 0208-1978
    #https://en.wikipedia.org/wiki/ISO/IEC_2022#ISO.2FIEC_2022_character_sets
    kanji = (b"\033$B" + data[1].to_bytes(2, byteorder="big")).decode("iso2022_jp").encode("utf-8")

    #convert bytes to image to numpy array
    iP = Image.frombytes("F", (128, 127), data[14], "bit", 4).convert("P")
    enhanced = ImageEnhance.Brightness(iP).enhance(16)
    arr = np.asarray(enhanced)
    pad = np.vstack([np.zeros(128), arr])

    return (pad, kanji)

def make_dataset(num_classes):
    #there are 200 characters of each class. 50 files * 4 writers per file
    #3036 classes
    chars = random.sample(range(0, 3036), num_classes)

    x = []
    y = []
    y_one = 0   #variable to convert string to onehot
    c_dict = {}
    for i in range(50):
        filename = "ETL9G/ETL9G_{0:02}".format(i+1) #i starts at 0, files start at 1
        for j in range(len(chars)):
            for k in range(4):
                loc = chars[j] + k*3036
                rec = retrieve(filename, loc)
                x.append(rec[0])
                #populate c_dict if first time through
                if i == 0:
                    if rec[1] in c_dict:
                        pass
                    else:
                        c_dict[rec[1]] = y_one
                        y_one += 1

                y.append(c_dict[rec[1]])

    x_train = np.asarray(x).astype(np.float32)
    y_train = np.asarray(y)
    #y_train = np.zeros((y.size, y.max()+1))    #one_hot
    return x_train, y_train