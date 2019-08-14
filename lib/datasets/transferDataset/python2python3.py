import pickle
import glob
import shutil
import ipdb
import random
st = ipdb.set_trace
import sys
import os
folder = os.path.join("/Users/mihirprabhudesai/Documents/projects/rebuttal_pnp_original_ashar/sample_front_back_data/2dData/",sys.argv[1],"trees")
p2folder = folder.replace("trees","trees_p2")
try:
	shutil.copytree(folder,p2folder)
except Exception:
	pass
tree_files = glob.glob(p2folder+"/train/*")
st()
for i in tree_files:
	val = pickle.load(open(i,"rb"))
	pickle.dump(val, open(i,"wb"), protocol=2)