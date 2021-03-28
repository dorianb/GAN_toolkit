# Generative Adversial Network toolkit

Python 3.7 has been chosen to ensure components compatibility. The project is structured using Distutils as a build 
automation tool. Test are implemented for each module using unittest.

## Environment configuration

#### Build the project's image
```
docker build -t gan_lab . 
```

#### Start container
```
docker run --gpus all -it --rm \
 --shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
 -v /home/dorian/workspace:/workspace \
 -p 4888:8888 \
 --name dorian_gan_lab \
 gan_lab 
```

## Installation procedure

Installation of third party librairies
```
$ pip install -r requirements.txt
```

Installation of modules
```
$ python setup.py install
```