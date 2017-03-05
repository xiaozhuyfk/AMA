# Read a file
# filename is the path of the file, string type
# returns the content as a string
def readFile(filename, mode="rt"):
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
def writeFile(filename, contents, mode="wt"):
    # wt stands for "write text"
    fout = None
    try:
        fout = open(filename, mode)
        fout.write(contents)
    finally:
        if (fout != None): fout.close()
    return True


content = "Universe = vanilla\n\n"
pattern = """executable = /home/hongyul/AMA/wiki.sh
arguments = %d
output = /home/hongyul/condor/condor%d.job.out
log = /tmp/hongyul%d.job.log
error = /home/hongyul/condor/condor%d.job.err
queue

"""

for i in xrange(200):
    idx = i
    test = pattern % (idx, idx, idx, idx)
    content += test

writeFile("wiki.job", content)
