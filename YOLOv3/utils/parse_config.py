"""
    这个py文件是用来解析读取网络模型.cfg文件和解析data的配置文件的
"""


def parse_model_config(path):
    # path:.cfg文件的路径
    file = open(path,'r')      # 读取文件
    lines = file.read().split('\n')     # 读取每一行数据
    lines = [x for x in lines if x and not x.startswith('#')]   # 去除每行是空的和以"#"开头的，startswith(str,beg=0,end=len(string))方法检查字符串是否以指定子字符串开头，如果是返回True，否则返回false，如果参数 beg 和 end 指定值，则在指定范围内检查。
    lines = [x.rstrip().lstrip() for x in lines]    # 去掉左边和右边都是空格
    #print(lines)
    module_defs = []        # 模型的结构
    for line in lines:      # 遍历.cfg文件里的每一行
        if line.startswith('['):     # 如果是以'['开头的行，则将该行的字符作为字典的键值
            module_defs.append({})      # 将每一个[]开头下面的数据都保存到字典中
            module_defs[-1]['type'] = line[1:-1].rstrip()   # 创建一个'type'的key，保存的是这一行的字符串，例如：[convolutional],最后返回的是[{'type':'convlutional'}.......]
            if module_defs[-1]['type'] == 'convolutional':      # 如果这一行是'[convolutional]',则在字典最后添加键值['batch_normalize']=0
                 module_defs[-1]['batch_normalize'] = 0
        else:
            key,value = line.split("=")     # 否则，将从等号拆分key和value
            value = value.strip()
            module_defs[-1][key.rstrip()] = value.strip()

    ##print("module_def",module_defs)
    return module_defs

# parse_model_config(r'E:\github代码\PyTorch-YOLOv3-master\config\yolov3.cfg')


def parse_data_config(path):
    """
        Parses the data configuration file,返回一个字典
    """
    options = dict()        # 定义一个空字典
    options['gpus'] = '0,1,2,3'     # 往字典里添加键key为‘gpus’，值为'0,1,2,3'
    options['num_workers'] = '10'   # 往字典里添加键key为'num_workers',值为'10'
    with open(path,'r') as fp:      # 打开coco.data文件夹，里面的内容是classes=80,train=data/coco/trainvalno5k.txt,valid=data/coco/5k.txt,names=data/coco.names
        for line in fp.readlines():
            line = line.strip()     # 去掉空格
            if line == ' ' or line.startswith('#'):     # 如果line等于空格或者是以#开头的，跳过
                continue
            key,val = line.split("=")       # 将等号两边分成key和value
            options[key.strip] = val.strip()
    return options



