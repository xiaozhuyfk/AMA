import codecs
import json

# Read a file
# filename is the path of the file, string type
# returns the content as a string
def readFile(filename, mode = "rt"):
    # rt stands for "read text"
    fin = contents = None
    try:
        fin = open(filename, mode)
        contents = fin.read()
    finally:
        if (fin != None): fin.close()
    return contents


# Write 'contents' to the file
# 'filename' is the path of the file, string type
# 'contents' is of string type
# returns True if the content has been written successfully
def writeFile(filename, contents, mode = "wt"):
    # wt stands for "write text"
    fout = None
    try:
        fout = open(filename, mode)
        fout.write(contents)
    finally:
        if (fout != None): fout.close()
    return True

def codecsReadFile(filename, mode = "rt", encoding = 'utf-8'):
    # rt stands for "read text"
    f = contents = None
    try:
        f = codecs.open(filename, mode=mode, encoding=encoding)
        contents = f.read()
    finally:
        if (f != None): f.close()
    return contents

def codecsWriteFile(filename, contents, mode = "wt", encoding = 'utf-8'):
    f = None
    try:
        f = codecs.open(filename, mode=mode, encoding=encoding)
        f.write(contents)
    finally:
        if (f != None): f.close()
    return True

def codecsLoadJson(filename, mode = "rt", encoding = 'utf-8'):
    f = None
    d = None
    try:
        with codecs.open(filename, mode, encoding) as f:
            d = json.load(f)
    finally:
        if (f != None): f.close()
    return d

def codecsDumpJson(filename, contents, mode = "wt", encoding = 'utf-8'):
    f = None
    try:
        with codecs.open(filename, mode, encoding) as f:
            json.dump(contents, f)
    finally:
        if (f != None): f.close()
    return True
