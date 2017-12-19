# im2txt_demo

im2txt (https://github.com/tensorflow/models/tree/master/research/im2txt)

### Getting started:

#### Install docker
https://docs.docker.com/engine/installation/

#### Download pretrained models:
Download https://drive.google.com/file/d/0Bw6m_66JSYLlRFVKQ2tGcUJaWjA/view
Copy all files to the folder im2txt_pretrained
#### build docker image
```
cd im2txt_demo
make build
```

##### start container
```
make run
```

##### build run_inference
```
./build_inference.sh
```
##### get caption for image
Sample of offline picture:
```
./process_image.sh imgs/bikes.jpg
```
or you could use url directly:
```
./process_image.sh PICTURE_URL
```

