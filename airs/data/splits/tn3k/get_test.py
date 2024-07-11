import os

def list_files_in_directory(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_list.append(file_path)
    return file_list

# 替换 'your_directory_path' 为你想要遍历的文件夹路径
directory_path = '/home/ai1007/airs/data/tn3k/leftImg/test/'
file_paths = list_files_in_directory(directory_path)
list = []
for path in file_paths:
    path = path.split('/')
    name = path[6] + '/' + path[7] + '/' + path[8]
    list.append(name)
with open('/home/ai1007/airs/data/splits/tn3k/test.txt', 'w') as file:
    for path in list:
        file.write(path + '\n')