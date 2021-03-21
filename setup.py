from distutils.core import setup

setup(
    name='GAN_toolkit',
    version='1.0',
    packages=[
        'fcGAN',
        'dcGAN'
    ],
    package_dir={
        'fcGAN': 'src/fcGAN',
        'dcGAN': 'src/dcGAN'
    },
    author='Dorian Bagur',
    author_email='dorian.bagur@gmail.com',
    description='Generative Adversial Network toolkit'
)