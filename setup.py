from distutils.core import setup

setup(
    name='GAN_toolkit',
    version='1.0',
    packages=[
        'fcGAN'
    ],
    package_dir={
        'fcGAN': 'src/fcGAN'
    },
    author='Dorian Bagur',
    author_email='dorian.bagur@gmail.com',
    description='Generative Adversial Network toolkit'
)