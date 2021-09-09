from setuptools import setup, find_packages

setup(
    name='metrics-lle', # 包名称，之后如果上传到了pypi，则需要通过该名称下载
    version='0.0.1', # version只能是数字，还有其他字符则会报错
    keywords=('metric', 'low light enhancement'),
    description='Metrics for low light enhancement',
    long_description='',
    license='MIT', # 遵循的协议
    install_requires=[
        'numpy<=1.19.5',
        'pillow<=8.2.0',
        'scipy<=1.7.1',
        'lpips<=0.1.3',
        'filetype<=1.0.7',
        'opencv-python<=4.5.3.56',
        'torch<=1.9.0',
        'torchvision<=0.10.0',
        'tqdm<=4.61.2',
        'scikit-image<=0.18.1'
    ], # 这里面填写项目用到的第三方依赖
    author='BreezeShane, PommesPeter, RuoMengAwA',
    author_email='bug.breeze.shane@gmail.com',
    packages=find_packages(), # 项目内所有自己编写的库
    platforms='any',
    url='git@github.com:LowLightTeam/Metrics.git', # 项目链接,
    include_package_data = True,
    entry_points={
        'console_scripts': [
        #     'example=run:main'
        ]
    },
)