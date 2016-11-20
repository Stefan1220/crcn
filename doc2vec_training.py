# coding:UTF-8
from gensim import corpora, models, similarities
import json
import os
import jieba

# from cpython cimport PyCObject_AsVoidPtr
# from scipy.linalg.blas import cblasfrom scipy.linalg.blas import cblas

# ctypedef void (*saxpy_ptr) (const int *N, const float *alpha, const float *X, const int *incX, float *Y, const int *incY) nogil
# cdef saxpy_ptr saxpy=<saxpy_ptr>PyCObject_AsVoidPtr(cblas.saxpy._cpointer)

jsonfile = open('./data/example_travel.json', 'r')
json_data=jsonfile.read()
jsondata=json.loads(json_data)
json_imgs=jsondata['images']
sentences=[]
for i,jsonimg in enumerate(json_imgs):
	concatpara=""
	for sentence in jsonimg['sentence']:
		ensent=sentence['raw']#.encode('ascii','ignore')
		if ensent not in concatpara:
			concatpara+=ensent
	key=str(i)
	concatpara_list = ' '.join(jieba.cut(concatpara)) #分词
	#print concatpara
	sentences.append(models.doc2vec.TaggedDocument(concatpara_list.split(' '), [key]))  #split默认以空格分隔，分出来的都是单词
	#print sentences
model = models.Doc2Vec(size=300,alpha=0.025, min_alpha=0.025,window=8, min_count=5, seed=1,sample=1e-5, workers=4)  # use fixed learning rate
model.build_vocab(sentences)
for epoch in range(100):
	print epoch
	model.train(sentences)
	model.alpha -= 0.0001  # decrease the learning rate
	model.min_alpha = model.alpha
	# if epoch%200==0 and epoch!=0:
	# 	print "save check point"
	# 	accuracy_list=model.accuracy('./model/questions-words.txt')
	# 	error=0
	# 	correct=0
	# 	for accuracy in accuracy_list:
	# 		error=error+len(accuracy['incorrect'])
	# 		correct=correct+len(accuracy['correct'])
	# 	print "accuracy :", correct*1.0/(correct+error)
	# 	model.save('./model/disney_model.doc2vec')
#model.init_sims(replace=True)

model.save('./model/example_travel.doc2vec')
