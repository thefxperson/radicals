import struct
from PIL import Image, ImageEnhance
 
filename = 'ETL9G/ETL9G_01'
id_record = 1
sz_record = 8199
with open(filename, 'rb') as f:
    f.seek(id_record * sz_record)
    s = f.read(sz_record)
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
    data = struct.unpack('>2H8sI4B4H2B34x8128s7x', s)
    print(data[0:14], hex(data[1]))

    #convert from bytes to a properly encoded string
    #it is unclear if character is encoded using the 1978 or 1983 JIS X 0208 standard. 83 is used here.
    #b"\033$B" is the ISO2022 escape code from JIS X 0208-1983. b"\033$@" is for JIX X 0208-1978
    #https://en.wikipedia.org/wiki/ISO/IEC_2022#ISO.2FIEC_2022_character_sets
    print((b"\033$B" + data[1].to_bytes(2, byteorder="big")).decode("iso2022_jp"))

    #convert bytes to image
    iF = Image.frombytes('F', (128, 127), data[14], 'bit', 4)
    iP = iF.convert('P')
    enhancer = ImageEnhance.Brightness(iP)
    iE = enhancer.enhance(16)

    #create the file name and save the image
    fn = 'ETL9G_{:d}_{:s}.png'.format((data[0]-1)%20+1, hex(int(data[1]))[-4:])
    iE.save(fn, 'PNG')