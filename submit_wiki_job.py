import subprocess

def SubmitOneJob(idx):
    lCmd = [
        '/bos/usr0/suruchis/Files/qsub',
        'python -m query_processor.rank_learner wiki %d' % idx
    ]
    CondorOut = subprocess.check_output(lCmd)
    print "submitted: [%s]\n out [%s]" % (json.dumps(lCmd), CondorOut)

for i in xrange(27):
    idx = i + 1
    SubmitOneJob(idx)