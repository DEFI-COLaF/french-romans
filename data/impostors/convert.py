import lxml.etree as et
import glob
import io
import shutil
import re

rename = re.compile("([0-9]+)_([a-zA-Z-ç.éÉèÉèÿé]+)(_.*)")

xsl = et.XSLT(et.fromstring("""<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
    xmlns:xs="http://www.w3.org/2001/XMLSchema"
    xmlns:math="http://www.w3.org/2005/xpath-functions/math"
    xpath-default-namespace="http://www.tei-c.org/ns/1.0"
    xmlns:tei="http://www.tei-c.org/ns/1.0"
    exclude-result-prefixes="xs math"
    version="1.0">
    <xsl:output method="text"/>
    
    <xsl:template match="tei:TEI">
        <xsl:apply-templates select=".//tei:body"/>
    </xsl:template>
    <xsl:template match="tei:note"/>
    <xsl:template match="tei:head"/>
</xsl:stylesheet>"""))

excluded_author = [
    "Balzac-Honore-de",
    "Dumas-Alexandre",
    "Sue-Eugene",
    "Sand-George",
    "Zola-Emile",
    "Verne-Jules"
]
for file in glob.glob("*.xml"):
    if file[0].isnumeric():
        print(file)
        date, author, *title = file.split("_")
        if author in excluded_author:
            continue
    with open(file) as f:
        data = f.read().replace("&nbsp;", " ")
    dest = rename.sub(r"\2_\1\3", file)
    print(dest)
    # shutil.move(file, dest)
    xml = et.parse(io.BytesIO(data.encode('utf-8')))
    string = str(xsl(xml))
    with open(dest.replace(".xml", ".txt"), "w") as f:
        f.write(string.strip())