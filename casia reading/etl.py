import struct
import numpy as np
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

    return (arr, kanji)


#variable to keep track of the writer of each character
writer_id = 0
#datasets
images = []
characters = []
writers = []

#ETL9G is split into 50 files
for i in range(50):
    filename = "ETL9G/ETL9G_{0:02}".format(i+1) #i starts at 0, files start at 1
    print(filename)

    #3036 kanji/writer, 4 writers/file. 3036*4=12144
    for record in range(12144):
        img, kanji = retrieve(filename, record) #read record from file

        #check to see if we have moved to a new writer
        if record % 3036 == 0:
            writer_id += 1

        images.append(img)
        characters.append(kanji)
        writers.append(writer_id)

#save data to single npz file for easy access.
np.savez("ETL9G.npz", images=np.asarray(images), characters=np.asarray(characters), writers=np.asarray(writers))