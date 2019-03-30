# text-detection-ctpn

***
# roadmap
- [x] reonstruct the repo
- [x] cython nms and bbox utils
- [x] loss function as referred in paper
- [x] oriented text connector
- [x] BLSTM
***
# setup
nms and bbox utils are written in cython, hence you have to build the library first.
```shell
cd utils/bbox
chmod +x make.sh
./make.sh
```

# 预处理图片
```shell
cd utils/prepare
python split_label.py
```

# 训练模型
```shell
cd main
python train.py
```

# 测试模型
```shell
cd main
python test.py
```

# web服务测试模型
```shell
python server.py
```