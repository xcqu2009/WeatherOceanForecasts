# TASK1_“AI Earth”人工智能创新挑战赛Docker提交操作步骤



## 一、创建自己的docker文件

```shell
# 1. 创建目录
mkdir WeatherOceanForecasts

# 2.编辑Dockerfile文件
vi Dockerfile
## 将以下文件内容复制到其中
## 从天池基础镜像构建
FROM registry.cn-shanghai.aliyuncs.com/tcc-public/tensorflow:latest-cuda10.0-py3

## 把当前文件夹里的文件构建到镜像的根目录下（.后面有空格，不能直接跟/）
ADD . /

## 指定默认工作目录为根目录（需要把run.sh和生成的结果文件都放在该文件夹下，提交后才能运行）
WORKDIR /

## Install Requirements（requirements.txt包含python包的版本）
## 这里使用清华镜像加速安装
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade pip
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
## 镜像启动后统一执行 sh run.sh
CMD ["sh", "run.sh"]

# 3.编辑requirements.txt文件
vi requirements.txt
# 将一下内容复制到文件中
numpy
tensorflow==2.2.0

# 4.创建code文件夹及python程序
mkdir code

vi mlp_predict.py
# 将一下内容复制到mlp_predict.py中

# 5.创建user_data文件夹及内容
mkdir -p user_data/model_data
# 将model_mlp_baseline.h5文件复制到目录下

# 6.创建result文件加载
mkdir result

# 7.创建run.sh文件

vi run.sh
# 将一下内容复制到run.sh文件中
#!/bin/sh
CURDIR="`dirname $0`" #获取此脚本所在目录
echo $CURDIR
cd $CURDIR #切换到该脚本所在目录
python /code/mlp_predict.py

# 7.build镜像文件
docker build -t registry.cn-shenzhen.aliyuncs.com/xcqu_for_tianchi/ai_earth_submit:1.0 .

# 8.pull 镜像文件
docker login --username=xcqu2020 registry.cn-shenzhen.aliyuncs.com
docker push registry.cn-shenzhen.aliyuncs.com/xcqu_for_tianchi/ai_earth_submit:1.0
```

- mlp_predict.py文件内容

```python
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import Input 
import numpy as np
import os
import zipfile

def RMSE(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

def build_model():  
    inp    = Input(shape=(12,24,72,4))  
    
    x_4    = Dense(1, activation='relu')(inp)   
    x_3    = Dense(1, activation='relu')(tf.reshape(x_4,[-1,12,24,72]))
    x_2    = Dense(1, activation='relu')(tf.reshape(x_3,[-1,12,24]))
    x_1    = Dense(1, activation='relu')(tf.reshape(x_2,[-1,12]))
     
    x = Dense(64, activation='relu')(x_1)  
    x = Dropout(0.25)(x) 
    x = Dense(32, activation='relu')(x)   
    x = Dropout(0.25)(x)  
    output = Dense(24, activation='linear')(x)   
    model  = Model(inputs=inp, outputs=output)

    adam = tf.optimizers.Adam(lr=1e-3,beta_1=0.99,beta_2 = 0.99) 
    model.compile(optimizer=adam, loss=RMSE)

    return model 

model = build_model()
model.load_weights('./user_data/model_data/model_mlp_baseline.h5')

test_path = './tcdata/enso_round1_test_20210201/'

### 1. 测试数据读取
files = os.listdir(test_path)
test_feas_dict = {}
for file in files:
    test_feas_dict[file] = np.load(test_path + file)
    
### 2. 结果预测
test_predicts_dict = {}
for file_name,val in test_feas_dict.items():
    test_predicts_dict[file_name] = model.predict(val).reshape(-1,)
#     test_predicts_dict[file_name] = model.predict(val.reshape([-1,12])[0,:])

### 3.存储预测结果
for file_name,val in test_predicts_dict.items(): 
    np.save('./result/' + file_name,val)

#打包目录为zip文件（未压缩）
def make_zip(source_dir='./result/', output_filename = 'result.zip'):
    zipf = zipfile.ZipFile(output_filename, 'w')
    pre_len = len(os.path.dirname(source_dir))
    source_dirs = os.walk(source_dir)
    print(source_dirs)
    for parent, dirnames, filenames in source_dirs:
        print(parent, dirnames)
        for filename in filenames:
            if '.npy' not in filename:
                continue
            pathfile = os.path.join(parent, filename)
            arcname = pathfile[pre_len:].strip(os.path.sep)   #相对路径
            zipf.write(pathfile, arcname)
    zipf.close()
make_zip()

```

- **疑惑之处**

  - 使用Dockerfile创建Docker镜像时，总是出现python安装包timeout的问题，在`pip install --timeout=600 `参数后仍然无法解决问题。后期采用，先创建基础image，在手工安装python包的方式解决问题。

  - 在运行完`From`命令后，出现了一个如下的image文件，不知道是什么原因。下图中标红色的地方
    ![image-20210218194518132](https://github.com/xcqu2009/WeatherOceanForecasts/blob/main/img/image-20210218194518132.png)
    
    

    







## 二、创建案例本地仓库

登录阿里云知识仓库:

https://cr.console.aliyun.com/repository/cn-shenzhen/xcqu_for_tianchi/ai_earth_submit/details

### 1. 登录阿里云Docker Registry

```shell
$ sudo docker login --username=xcqu2020 registry.cn-shenzhen.aliyuncs.com
```

用于登录的用户名为阿里云账号全名，密码为开通服务时设置的密码。

您可以在访问凭证页面修改凭证密码。

### 2. 从Registry中拉取镜像

```shell
$ sudo docker pull registry.cn-shenzhen.aliyuncs.com/xcqu_for_tianchi/ai_earth_submit:[镜像版本号]
```

### 3. 将镜像推送到Registry

```shell
$ sudo docker login --username=xcqu2020 registry.cn-shenzhen.aliyuncs.com
$ sudo docker tag [ImageId] registry.cn-shenzhen.aliyuncs.com/xcqu_for_tianchi/ai_earth_submit:[镜像版本号]
$ sudo docker push registry.cn-shenzhen.aliyuncs.com/xcqu_for_tianchi/ai_earth_submit:[镜像版本号]
```

请根据实际镜像信息替换示例中的[ImageId]和[镜像版本号]参数。

### 4. 选择合适的镜像仓库地址

从ECS推送镜像时，可以选择使用镜像仓库内网地址。推送速度将得到提升并且将不会损耗您的公网流量。

如果您使用的机器位于VPC网络，请使用 registry-vpc.cn-shenzhen.aliyuncs.com 作为Registry的域名登录。

### 5. 示例

使用"docker tag"命令重命名镜像，并将它通过专有网络地址推送至Registry。

```shell
$ sudo docker imagesREPOSITORY                                                         TAG                 IMAGE ID            CREATED             VIRTUAL SIZEregistry.aliyuncs.com/acs/agent                                    0.7-dfb6816         37bb9c63c8b2        7 days ago          37.89 MB$ sudo docker tag 37bb9c63c8b2 registry-vpc.cn-shenzhen.aliyuncs.com/acs/agent:0.7-dfb6816
```

使用 "docker push" 命令将该镜像推送至远程。

```shell
$ sudo docker push registry-vpc.cn-shenzhen.aliyuncs.com/acs/agent:0.7-dfb6816
```











## 三、提交镜像

```shell
# 通过修改后commit镜像变更
docker commit -m "Update docker images" registry.cn-shenzhen.aliyuncs.com/xcqu_for_tianchi/ai_earth_submit:1.0  c5d106f46ae7

# 提交到本地仓库
docker login --username=xcqu2020 registry.cn-shenzhen.aliyuncs.com
docker push registry.cn-shenzhen.aliyuncs.com/xcqu_for_tianchi/ai_earth_submit:1.0

```

提交镜像后一致显处于waiting状态，经过查看比赛文档说明为镜像仓库用户、密码、地址错误导致，更新后提示没有问题。

![image-20210218193741428](https://github.com/xcqu2009/WeatherOceanForecasts/blob/main/img/image-20210218193741428.png)

![image-20210218210739468](https://github.com/xcqu2009/WeatherOceanForecasts/blob/main/img/image-20210218210739468.png)

