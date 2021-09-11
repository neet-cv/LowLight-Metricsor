from setuptools import setup, find_packages

setup(
    name='metrics-lle', # 包名称，之后如果上传到了pypi，则需要通过该名称下载
    version='0.0.3', # version只能是数字，还有其他字符则会报错
    keywords=['metric', 'low light enhancement'],
    description='Metrics for low light enhancement',
    long_description='',
    license='MIT', # 遵循的协议
    install_requires=[
        'numpy',
        'pillow',
        'scipy',
        'lpips',
        'filetype',
        'opencv-python',
        'torch',
        'torchvision',
        'tqdm',
        'scikit-image'
    ],  # 这里面填写项目用到的第三方依赖
    author='BreezeShane, PommesPeter, RuoMengAwA',
    author_email='bug.breeze.shane@gmail.com',
    packages=find_packages(), # 项目内所有自己编写的库
    platforms='any',
    url='https://github.com/LowLightTeam/Metrics', # 项目链接,
    include_package_data=True,
    entry_points={
        'console_scripts': [
        #     'example=run:main'
        ]
    },
)