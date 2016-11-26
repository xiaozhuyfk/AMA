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
            json.dump(contents, f, indent=4)
    finally:
        if (f != None): f.close()
    return True

"""return a tuple with recall, precision, and f1 for one example"""
def computeF1(goldList,predictedList):

  """Assume all questions have at least one answer"""
  if len(goldList)==0:
      if len(predictedList)==0:
          return (1,1,1)
      else:
          return (0,0,0)
  """If we return an empty list recall is zero and precision is one"""
  if len(predictedList)==0:
      return (0,1,0)
  """It is guaranteed now that both lists are not empty"""

  precision = 0
  for entity in predictedList:
      if entity in goldList:
          precision+=1
  precision = float(precision) / len(predictedList)

  recall=0
  for entity in goldList:
      if entity in predictedList:
          recall+=1
  recall = float(recall) / len(goldList)

  f1 = 0
  if precision+recall>0:
      f1 = 2*recall*precision / (precision + recall)
  return (recall,precision,f1)

import subprocess
def kstem(stem):
    cmd = ['java',
           '-classpath',
           'kstem.jar',
           'org.lemurproject.kstem.KrovetzStemmer',
           '-w',
           stem]
    p = subprocess.Popen(cmd, stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
    out, err = p.communicate()
    result = out.split(" ")[1][:-1]
    return result


if __name__ == '__main__':
    #print edit_distance('this is a house', 'this is not a house')
    #sftp_get("/home/hongyul/Python-2.7.11.tgz", "/Users/Hongyu1/Desktop/Python.tgz")
    #sftp_get_r("/home/hongyul/query", "/Users/Hongyu1/Desktop")
    #sftp_put("/Users/Hongyu1/Desktop/Python.tgz", "/home/hongyul/haha.tgz")
    #print sftp_execute("../init_env/bin/python indri.py name_of_collection_activity")
    #print sftp_listdir("/home/hongyul/")
    #get_filenames()
    #sftp_put("/data/dump.tar.gz", "/home/hongyul/aqqu/testresult/dump")
    #test()
    print kstem("what-if")
    print kstem("Paris")
    print kstem("imaging")
    print kstem("discovered")
    print kstem("looping")