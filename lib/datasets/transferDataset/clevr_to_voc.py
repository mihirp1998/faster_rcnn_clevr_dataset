
import sys
import argparse
from xml.dom.minidom import Document
import cv2, os
import glob
import xml.etree.ElementTree as ET
import shutil
import numpy as np
import ipdb
import pickle
st = ipdb.set_trace
import copy
import random
diff_class = False
import imageio
from scipy.misc import imsave
def append_xml_node_attr(child, parent = None, text = None,doc=None):
	ele = doc.createElement(child)
	if not text is None:
		text_node = doc.createTextNode(text)
		ele.appendChild(text_node)
	parent = doc if parent is None else parent
	parent.appendChild(ele)
	return ele

def generate_xml(name,tree, img_size = (370, 1224, 3), \
				 class_sets = ('pedestrian', 'car', 'cyclist'), \
				 doncateothers = True):
	"""
	Write annotations into voc xml format.
	Examples:
		In: 0000001.txt
			cls        truncated    occlusion   angle   boxes                         3d annotation...
			Pedestrian 0.00         0           -0.20   712.40 143.00 810.73 307.92   1.89 0.48 1.20 1.84 1.47 8.41 0.01
		Out: 0000001.xml
			<annotation>
				<folder>VOC2007</folder>
				<filename>000001.jpg</filename>
				<source>
				...
				<object>
					<name>Pedestrian</name>
					<pose>Left</pose>
					<truncated>1</truncated>
					<difficult>0</difficult>
					<bndbox>
						<xmin>x1</xmin>
						<ymin>y1</ymin>
						<xmax>x2</xmax>
						<ymax>y2</ymax>
					</bndbox>
				</object>
			</annotation>
	:param name: stem name of an image, example: 0000001
	:param lines: lines in kitti annotation txt
	:param img_size: [height, width, channle]
	:param class_sets: ('Pedestrian', 'Car', 'Cyclist')
	:return:
	"""

	doc = Document()


	img_name = name+'.jpg'

	# create header
	annotation = append_xml_node_attr('annotation',doc=doc)
	append_xml_node_attr('folder', parent = annotation, text='KITTI',doc=doc)
	append_xml_node_attr('filename', parent = annotation, text=img_name,doc=doc)
	source = append_xml_node_attr('source', parent=annotation,doc=doc)
	append_xml_node_attr('database', parent=source, text='KITTI',doc=doc)
	append_xml_node_attr('annotation', parent=source, text='KITTI',doc=doc)
	append_xml_node_attr('image', parent=source, text='KITTI',doc=doc)
	append_xml_node_attr('flickrid', parent=source, text='000000',doc=doc)
	owner = append_xml_node_attr('owner', parent=annotation,doc=doc)
	append_xml_node_attr('url', parent=owner, text = 'http://www.cvlibs.net/datasets/kitti/index.php',doc=doc)
	size = append_xml_node_attr('size', annotation,doc=doc)
	append_xml_node_attr('width', size, str(img_size[1]),doc=doc)
	append_xml_node_attr('height', size, str(img_size[0]),doc=doc)
	append_xml_node_attr('depth', size, str(img_size[2]),doc=doc)
	append_xml_node_attr('segmented', parent=annotation, text='0',doc=doc)

	# create objects
	objs = []
	# for line in lines:
	# if tree.function == "layout":
	# 	print("hello")
	# 	st()
	objs = compose_tree(tree,objs,doc,annotation)
	# for child in tree.children:
	# 	assert child.function == "describe"
	# 	cls = child.word
	# 	if not doncateothers and cls not in class_sets:
	# 		continue
	# 	cls = 'dontcare' if cls not in class_sets else cls
	# 	obj = append_xml_node_attr('object', parent=annotation,doc=doc)
	# 	occlusion = 0

	# 	x1,y1,h,w = child.bbox
	# 	x2,y2 = (x1+h,y1+w)

	# 	# x1, y1, x2, y2 = int(float(splitted_line[4]) + 1), int(float(splitted_line[5]) + 1), \
	# 	# 				 int(float(splitted_line[6]) + 1), int(float(splitted_line[7]) + 1)
	# 	# truncation = float(splitted_line[1])
	# 	truncation= 0.00
	# 	# difficult = 1 if _is_hard(cls, truncation, occlusion, x1, y1, x2, y2) else 0
	# 	# truncted = 0 if truncation < 0.5 else 1
	# 	truncted = 0
	# 	difficult = 0
	# 	append_xml_node_attr('name', parent=obj, text=cls,doc=doc)
	# 	append_xml_node_attr('pose', parent=obj, text='Left',doc=doc)
	# 	append_xml_node_attr('truncated', parent=obj, text=str(truncted),doc=doc)
	# 	append_xml_node_attr('difficult', parent=obj, text=str(int(difficult)),doc=doc)
	# 	bb = append_xml_node_attr('bndbox', parent=obj,doc=doc)
	# 	append_xml_node_attr('xmin', parent=bb, text=str(x1),doc=doc)
	# 	append_xml_node_attr('ymin', parent=bb, text=str(y1),doc=doc)
	# 	append_xml_node_attr('xmax', parent=bb, text=str(x2),doc=doc)
	# 	append_xml_node_attr('ymax', parent=bb, text=str(y2),doc=doc)

	# 	o = {'class': cls, 'box': np.asarray([x1, y1, x2, y2], dtype=float), \
	# 		 'truncation': truncation, 'difficult': difficult, 'occlusion': occlusion}
	# 	objs.append(o)

	return  doc, objs

def compose_tree(treex, objs,doc,annotation):
	for i in range(0, treex.num_children):
		objs = compose_tree(treex.children[i],objs,doc,annotation)
	if treex.function == "describe":
		if diff_class:
			cls = treex.word
		else:
			cls = 'single'


		cls = 'dontcare' if cls not in class_sets else cls
		obj = append_xml_node_attr('object', parent=annotation,doc=doc)
		occlusion = 0

		x1,y1,h,w = treex.bbox
		x2,y2 = (x1+h,y1+w)
		if x2 > 64:
			x2 = 63
			# st()
		if x1 > 64:
			x1=63
			# st()
		# if x1 > x2:
		# 	st()

		# x1, y1, x2, y2 = int(float(splitted_line[4]) + 1), int(float(splitted_line[5]) + 1), \
		# 				 int(float(splitted_line[6]) + 1), int(float(splitted_line[7]) + 1)
		# truncation = float(splitted_line[1])
		truncation= 0.00
		# difficult = 1 if _is_hard(cls, truncation, occlusion, x1, y1, x2, y2) else 0
		# truncted = 0 if truncation < 0.5 else 1
		truncted = 0
		difficult = 0
		append_xml_node_attr('name', parent=obj, text=cls,doc=doc)
		append_xml_node_attr('pose', parent=obj, text='Left',doc=doc)
		append_xml_node_attr('truncated', parent=obj, text=str(truncted),doc=doc)
		append_xml_node_attr('difficult', parent=obj, text=str(int(difficult)),doc=doc)
		bb = append_xml_node_attr('bndbox', parent=obj,doc=doc)
		append_xml_node_attr('xmin', parent=bb, text=str(x1),doc=doc)
		append_xml_node_attr('ymin', parent=bb, text=str(y1),doc=doc)
		append_xml_node_attr('xmax', parent=bb, text=str(x2),doc=doc)
		append_xml_node_attr('ymax', parent=bb, text=str(y2),doc=doc)

		o = {'class': cls, 'box': np.asarray([x1, y1, x2, y2], dtype=float), \
			 'truncation': truncation, 'difficult': difficult, 'occlusion': occlusion}
		objs.append(o)
	return objs

def _is_hard(cls, truncation, occlusion, x1, y1, x2, y2):
	# Easy: Min. bounding box height: 40 Px, Max. occlusion level: Fully visible, Max. truncation: 15 %
	# Moderate: Min. bounding box height: 25 Px, Max. occlusion level: Partly occluded, Max. truncation: 30 %
	# Hard: Min. bounding box height: 25 Px, Max. occlusion level: Difficult to see, Max. truncation: 50 %
	hard = False
	if y2 - y1 < 25 and occlusion >= 2:
		hard = True
		return hard
	if occlusion >= 3:
		hard = True
		return hard
	if truncation > 0.8:
		hard = True
		return hard
	return hard

def parse_args():
	"""
	Parse input arguments
	"""
	parser = argparse.ArgumentParser(description='Convert clevr dataset into Pascal voc format')
	parser.add_argument('--clevr_dir', dest='clevr_dir',
						help='which folder to load',
						default='CLEVR_64_36_MORE_OBJ_FINAL_2D', type=str)
	parser.add_argument('--mode', dest='mode',
					help='which to load',
					default='test', type=str)
	parser.add_argument('--out', dest='outdir',
						help='path to voc-kitti',
						default='./data/CLEVR_MORE_OBJ', type=str)
	parser.add_argument('--draw', dest='draw',
						help='draw rects on images',
						default=0, type=int)
	parser.add_argument('--dontcareothers', dest='dontcareothers',
						help='ignore other categories, add them to dontcare rsgions',
						default=1, type=int)

	if len(sys.argv) == 1:
		parser.print_help()
		# sys.exit(1)
	args = parser.parse_args()
	return args

def _draw_on_image(img, objs, class_sets_dict):
	colors = [(86, 0, 240), (173, 225, 61), (54, 137, 255),\
			  (151, 0, 255)]
	font = cv2.FONT_HERSHEY_SIMPLEX
	old_img = copy.deepcopy(img)
	# cyan is sphere....red is cylinder... orange is cube
	for ind, obj in enumerate(objs):
		if obj['box'] is None: continue
		x1, y1, x2, y2 = obj['box'].astype(int)
		cls_id = class_sets_dict[obj['class']]
		if obj['class'] == 'dontcare':
			cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)
			continue
		cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), colors[cls_id % len(colors)], 1)
		# text = '{:s}*|'.format(obj['class'][:3]) if obj['difficult'] == 1 else '{:s}|'.format(obj['class'][:3])
		# text += '{:.1f}|'.format(obj['truncation'])
		# text += str(obj['occlusion'])
		# cv2.putText(img, text, (x1-2, y2-2), font, 0.5, (255, 0, 255), 1)
	return np.concatenate([img,old_img],1)

def build_voc_dirs(outdir):
	"""
	Build voc dir structure:
		VOC2007
			|-- Annotations
					|-- ***.xml
			|-- ImageSets
					|-- Layout
							|-- [test|train|trainval|val].txt
					|-- Main
							|-- class_[test|train|trainval|val].txt
					|-- Segmentation
							|-- [test|train|trainval|val].txt
			|-- JPEGImages
					|-- ***.jpg
			|-- SegmentationClass
					[empty]
			|-- SegmentationObject
					[empty]
	"""
	mkdir = lambda dir: os.makedirs(dir) if not os.path.exists(dir) else None
	mkdir(outdir)
	mkdir(os.path.join(outdir, 'Annotations'))
	mkdir(os.path.join(outdir, 'ImageSets'))
	mkdir(os.path.join(outdir, 'ImageSets', 'Layout'))
	mkdir(os.path.join(outdir, 'ImageSets', 'Main'))
	mkdir(os.path.join(outdir, 'ImageSets', 'Segmentation'))
	mkdir(os.path.join(outdir, 'JPEGImages'))
	mkdir(os.path.join(outdir, 'DepthImages'))

	mkdir(os.path.join(outdir, 'SegmentationClass'))
	mkdir(os.path.join(outdir, 'SegmentationObject'))

	return os.path.join(outdir, 'Annotations'), os.path.join(outdir, 'JPEGImages'), os.path.join(outdir, 'DepthImages'), os.path.join(outdir, 'ImageSets', 'Main')




if __name__ == '__main__':
	args = parse_args()

	_clevr = args.clevr_dir
	_mode = args.mode

	_clevrimgpath = os.path.join("../pnp_inside/data/CLEVR/" , _clevr,'images',_mode)

	_clevrtreepath = _clevrimgpath.replace("images","trees")
	# st()

	_clevr_img_files = glob.glob(_clevrimgpath+"/*")
	_clevr_img_files.sort()
	_clevr_tree_files = glob.glob(_clevrtreepath+"/*")
	_clevr_tree_files.sort()

	_outdir = args.outdir
	_draw = bool(args.draw)
	_dest_label_dir, _dest_img_dir, _dest_depth_dir, _dest_set_dir = build_voc_dirs(_outdir)
	_doncateothers = bool(args.dontcareothers)
	# for kitti only provides training labels
	for dset in [_mode]:

		# _labeldir = os.path.join(_kittidir, 'training', 'label_2')
		# _imagedir = os.path.join(_kittidir, 'training', 'image_2')
		"""
		class_sets = ('pedestrian', 'cyclist', 'car', 'person_sitting', 'van', 'truck', 'tram', 'misc', 'dontcare')
		"""
		if diff_class:
			class_sets = ('cylinder', 'sphere', 'cube')
		else:
			class_sets = ['single']
		# st()
		class_sets_dict = dict((k, i) for i, k in enumerate(class_sets))
		allclasses = {}
		fs = [open(os.path.join(_dest_set_dir, cls + '_' + dset + '.txt'), 'w') for cls in class_sets ]
		ftrain = open(os.path.join(_dest_set_dir, dset + '.txt'), 'w')
		# st()
		fval = open(os.path.join(_dest_set_dir, 'val' + '.txt'), 'w')
		# st()
		# files = glob.glob(os.path.join(_labeldir, '*.txt'))
		# files.sort()
		# st()
		for i,tree_file in enumerate(_clevr_tree_files):
			img_file = _clevr_img_files[i]
			stem = tree_file.split("/")[-1][:-5]
			stem_img = img_file.split("/")[-1][:-4]
			assert stem_img == stem

			with open(tree_file, 'rb') as f:
				tree  = pickle.load(f)
			# img_file = os.path.join(_imagedir, stem + '.png')
			depth_file = img_file.replace("images","depth").replace("png","exr")
			# st()
			depth = np.array(imageio.imread(depth_file, format='EXR-FI'))[:,:,0]
			depth = depth * (100 - 0) + 0
			depth.astype(np.float32)			
			img = cv2.imread(img_file)
			img_size = img.shape

			# st()
			doc, objs = generate_xml(stem,tree, img_size, class_sets=class_sets, doncateothers=_doncateothers)
			if _draw:
				val = _draw_on_image(img, objs, class_sets_dict)
				cv2.imwrite(os.path.join('demo.jpg'), val)
				st()
			# st()
			cv2.imwrite(os.path.join(_dest_img_dir, stem + '.jpg'), img)
			imsave(os.path.join(_dest_depth_dir, stem + '.jpg'), depth)

			xmlfile = os.path.join(_dest_label_dir, stem + '.xml')
			with open(xmlfile, 'w') as f:
				f.write(doc.toprettyxml(indent='	'))
			if random.random() <0.9:
				ftrain.writelines(stem + '\n')
			else:
				fval.writelines(stem + '\n')


			# build [cls_train.txt]
			# Car_train.txt: 0000xxx [1 | -1]
			cls_in_image = set([o['class'] for o in objs])

			for obj in objs:
				cls = obj['class']
				allclasses[cls] = 0 \
					if not cls in allclasses.keys() else allclasses[cls] + 1

			for cls in cls_in_image:
				if cls in class_sets:
					fs[class_sets_dict[cls]].writelines(stem + ' 1\n')
			for cls in class_sets:
				if cls not in cls_in_image:
					fs[class_sets_dict[cls]].writelines(stem + ' -1\n')

			# if int(stem) % 100 == 0:
			print(tree_file)

		(f.close() for f in fs)
		ftrain.close()
		fval.close()

		print('~~~~~~~~~~~~~~~~~~~')
		print(allclasses)
		print('~~~~~~~~~~~~~~~~~~~')
		# shutil.copyfile(os.path.join(_dest_set_dir, 'train.txt'), os.path.join(_dest_set_dir, 'val.txt'))
		shutil.copyfile(os.path.join(_dest_set_dir, '{}.txt'.format(dset)), os.path.join(_dest_set_dir, 'trainval.txt'))
		# shutil.copyfile(os.path.join(_dest_set_dir, '{}.txt'.format(dset)), os.path.join(_dest_set_dir, 'val.txt'))

		# for cls in class_sets:
		# 	shutil.copyfile(os.path.join(_dest_set_dir, cls + '_train.txt'),
		# 					os.path.join(_dest_set_dir, cls + '_trainval.txt'))
		# 	shutil.copyfile(os.path.join(_dest_set_dir, cls + '_train.txt'),
		# 					os.path.join(_dest_set_dir, cls + '_val.txt'))

