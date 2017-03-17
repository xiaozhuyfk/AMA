import plotly.plotly as py
from plotly.graph_objs import *

import codecs
import json
import subprocess


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


def codecsReadFile(filename, mode="rt", encoding='utf-8'):
    # rt stands for "read text"
    f = contents = None
    try:
        f = codecs.open(filename, mode=mode, encoding=encoding)
        contents = f.read()
    finally:
        if (f != None): f.close()
    return contents


def codecsWriteFile(filename, contents, mode="wt", encoding='utf-8'):
    f = None
    try:
        f = codecs.open(filename, mode=mode, encoding=encoding)
        f.write(contents)
    finally:
        if (f != None): f.close()
    return True


######################################################################

def plot_support_sentence(idx):
    path = "support_sentence_stat/webquestionstest/" + str(idx)
    lines = codecsReadFile(path).strip().split("\n")
    queries = [line.strip().split('\t')[0] for line in lines]
    support_count = [int(line.strip().split('\t')[1]) for line in lines]
    candidate_graph = [line.strip().split('\t')[2] for line in lines]
    candidate_support_count = [int(line.strip().split('\t')[3]) for line in lines]

    trace2 = Bar(
        x = candidate_graph,
        y = candidate_support_count,
        name='answer support sentence count'
    )

    data = [trace2]

    # Plot and embed in ipython notebook!
    layout = dict(
        title = queries[0],
        xaxis = dict(
            tickangle = 45,
            tickfont = dict(
                size = 10
            )
        ),
        yaxis = dict(
            title = "count"
        ),
        barmode = 'group',
        margin = dict(
            b = 200
        ),
    )

    fig = dict(
        data = data,
        layout = layout
    )
    #py.plot(fig, filename="support sentence count")

    py.image.save_as(fig, filename='support_sentence_stat/webquestionstest/' + str(idx) + ".png")


def main():
    """
    for i in xrange(2032):
        try:
            idx = i+1
            print idx
            plot_support_sentence(idx)
        except:
            pass
    """
    plot_support_sentence(100)
    #plot_support_sentence("support_sentence_stats_webquestionstest.txt")



if __name__ == '__main__':
    main()