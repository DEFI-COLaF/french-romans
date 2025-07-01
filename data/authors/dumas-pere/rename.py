import glob
import shutil
import re

rename = re.compile("([0-9]+)_([a-z]+)_(.*)")

for file in glob.glob("*.xml"):
    if rename.findall(file):
        dest = rename.sub(r"\2_\1_\3", file)
        shutil.move(file, dest)
        shutil.move(file.replace(".xml", ".txt"), dest.replace(".xml", ".txt"))