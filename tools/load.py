import glob
import os.path as op
import dataclasses
from typing import List
import unicodedata
import re


@dataclasses.dataclass
class Text:
	author: str
	date: int
	title: str
	text: str


_datapath = op.abspath(op.join(op.dirname(__file__), "..", "data"))



def normalize(string: str) -> str:
	""" Normalizes whitespace, except for newlines
	"""
	return re.sub(r"[^\S\r\n]+", " ", unicodedata.normalize("NFKC", string))


def load_authors() -> List[Text]:
	data = []
	for file in glob.glob(f"{_datapath}/authors/*/*.txt"):
		au, da, ti = op.basename(file).replace(".txt", "").split("_")
		with open(file, encoding="utf8", errors='ignore') as f:
			data.append(
				Text(au, int(da), ti, normalize(f.read()))
			)
	return data

print(load_authors()[0])