import os
from PIL import Image

def crop_img(image,img_name,im_type="img",split_num=2):
	if im_type=="img":
		save_path = img_split_path
	else:
		save_path = anno_split_path
	w,h = image.size  
	if split_num==4:
		box1 = (0,0,w//2,h//2) #左上角建系，需要给出对角的两个点的坐标 
		box2 = (w//2,0,w,h//2) 
		box3 = (0,h//2,w//2,h)
		box4 = (w//2,h//2,w,h)
		boxs = [box1,box2,box3,box4]
	elif split_num==2:
		box1 = (0,0,w//2,h)
		box2 = (w//2,0,w,h)
		boxs = [box1,box2]

	for num,box in enumerate(boxs):
		im = image.crop(box)
		img_name_list = img_name.split(".")
		pre_save_path=os.path.join(save_path,img_name_list[0]+"_"+str(num+1)+"."+img_name_list[1])
		im.save(pre_save_path)
		print("save:{}/{}".format(i,len(img_path_list)),flush=True,end='\r')

def get_txt(img_folder,anno_folder):
	img_list=sorted(os.listdir(img_folder))
	anno_list=sorted(os.listdir(anno_folder))
	with open(split_save_txt,"w") as fw:
		for img_name,anno_name in zip(img_list,anno_list):
			img_path=os.path.join(img_folder,img_name)
			anno_path=os.path.join(anno_folder,anno_name)
			line=img_path+" "+anno_path+"\n"
			fw.write(line)
	print("txt gene done!")

if __name__=="__main__":
	split_data_txt = "../txt/camvid_ac_train.txt"
	split_save_txt = "../txt/camvid_split_train.txt"
	img_split_path = "/mnt/camvid_split/split_train_img/"
	anno_split_path = "/mnt/camvid_split/split_train_anno/" 
	if not os.path.exists(img_split_path):
		os.makedirs(img_split_path)
	if not os.path.exists(anno_split_path):
		os.makedirs(anno_split_path)

	with open(split_data_txt,"r") as f:
		lines = [line.strip().split(" ") for line in f]
		img_path_list=[lines[i][0] for i in range(len(lines))]
		label_path_list=[lines[i][1] for i in range(len(lines))]

	i=0
	for img_path,label_path in zip(img_path_list,label_path_list):
		i+=1
		image = Image.open(img_path) 
		crop_img(image,img_path.split("/")[-1])
		label= Image.open(label_path) 
		crop_img(label,label_path.split("/")[-1],im_type="anno")
	get_txt(img_split_path,anno_split_path)
