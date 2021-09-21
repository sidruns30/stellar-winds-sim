
import glob
import os
import sys


def compare_dumps():
	list = glob.glob("%s*" %dump_folder1)
	for dump_total in list:
		dump_name = dump_total[67:]
		os.system("cmp %s%s %s%s" % (dump_folder1,dump_name, dump_folder2,dump_name))
		print (dump_name)



if __name__ == "__main__":
	dump_folder1 = sys.argv[1]
	dump_folder2 = sys.argv[2]
	compare_dumps()


