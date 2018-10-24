test_file_dict={}

with open('/home/rtavaron/KaldiTimit/data/coretest_PickleFile_indeces.txt') as fp:
	for line in fp:
		value,key = line.split('\t')
		key=key.replace('\n','')
		value = value.replace('/home/local/IIT/lbadino/Data/KP/timit/test/','').replace('.npy','')
		test_file_dict[key]=value
