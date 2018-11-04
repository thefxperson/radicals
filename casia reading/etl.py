import struct
import tensorflow as tf
from PIL import Image

num_records = 12144
record_size = 8199

characters = []
images = []

for file_num in range(50):
    filename = "ETL9G/ETL9G_" + str(file_num+1).zfill(2)
    with open(filename, "r") as file:
        for record in range(num_records):
            file.seek(record*record_size)
            data = struct.unpack(">2H8sI4B4H2B34x8128s7x", file.read(record_size).encode("utf_8"))      #see ETL-9B specification
            characters.append((b"\033$B" + data[1]).decode("iso2022_jp").encode("utf-8"))    #0b\033$B is an escape character which allows us to use JIS x 0208 (83) with ISO2022. Store as Unicode
            images.append(Image.frombytes("F", (128, 127), data[14], "bit", 4).convert("P"))     #read image as floating, then convert to ints. 4 bits per pixel
