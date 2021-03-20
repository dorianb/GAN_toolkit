from distutils.core import setup

setup(
    name='GAN_toolkit',
    version='1.0',
    packages=[
        'simpleGAN'
    ],
    package_dir={
        'simpleGAN': 'src/simpleGAN'
    },
    author='Dorian Bagur',
    author_email='dorian.bagur@gmail.com',
    description='Generative Adversial Network toolkit'
)