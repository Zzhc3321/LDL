import os
from matplotlib import pyplot as plt
import re
source_folder = "FGNET/"

points_file_names = os.listdir(source_folder+"points")
images_file_names = os.listdir(source_folder+"images")
test_name = str.lower(images_file_names[2])

with open(source_folder+'points/'+test_name[0:6]+'.pts','r') as f:
    points = f.readlines()[3:-2]

x_s = []
y_s = []
for point in points:
    retx = re.match("[0-9]*[.][0-9]*",point)
    rety = re.match("[0-9]*[.][0-9]*",point.replace(retx.group()+' ',''))
    x_s.append(float(retx.group()))
    y_s.append(float(rety.group()))

img = plt.imread(source_folder+"images/"+test_name)

print(img)
print(img.shape)
plt.imshow(img,cmap='gray')
plt.plot(x_s,y_s)
plt.show()

f=open(r'C:\Users\赵华众\PycharmProjects\hello python\LDL\年龄估计\FGNET\Data_files\004a19.dat',encoding="UTF8")
print(f.read())