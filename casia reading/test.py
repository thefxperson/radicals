class sample():
	def __init__(self):
		pass

	def process(self, file):
		self.size = int.from_bytes(file.read(4), byteorder="big", signed=False)
		code = file.read(2)
		if code[1] == 0:
			self.char = chr(code[0])
		else:
			self.char = code.decode("gb2312")
		self.width = int.from_bytes(file.read(2), byteorder="big", signed=False)
		self.height = int.from_bytes(file.read(2), byteorder="big", signed=False)

with open("001-f.gnt", "rb") as file:
	t = sample()
	t.process(file)
	print(t.size, t.char, t.width, t.height)
	print(t.width * t.height)