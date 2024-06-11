from setuptools import setup, find_packages
import os


def read_long_description():
    root = os.path.abspath(os.path.dirname(__file__))
    path = os.path.join(root, "README.md")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    return text


setup(
    name='simple-diffusion',
    packages=find_packages(exclude=['notebooks']),
    version='0.2.0',
    license='MIT',
    description=
    'A minimal implementation of diffusion with a DDIM scheduler in PyTorch',
    author='Filip Basara',
    author_email='basarafilip@gmail.com',
    url='https://github.com/filipbasara0/simple-diffusion',
    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    keywords=[
        'machine learning', 'pytorch', 'self-supervised learning',
        'representation learning', 'contrastive learning'
    ],
    install_requires=[
        'torch>=2.1', 'torchvision>=0.16', 'transformers>=4.35',
        'datasets>=2.15', 'tqdm>=4.66', 'torchinfo>=1.8.0',
        'matplotlib>=3.6.0', 'diffusers>=0.24.0', 'einops>=0.7.0',
        'webdataset>=0.2.77',
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
    ],
    entry_points={
        "console_scripts": ["train_diffusion = train:main"],
    },
)
