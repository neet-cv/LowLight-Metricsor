# Metrcis for Low Light Enhancement

Here we provide the project able to compute metrics for low light enhancement !

Metrics provided are these:

- MAE
- MSE
- PSNR
- SSIM
- LPIPS
- NIQE
- NIMA

To use it, you need to prepared these packages:

```yaml
numpy~=1.19.5
pillow~=8.2.0
scipy~=1.7.1
lpips~=0.1.3
filetype~=1.0.7
opencv-python~=4.5.3.56
torch~=1.9.0
torchvision~=0.10.0
tqdm~=4.61.2
scikit-image~=0.18.1
```

We have proposed requirements.txt so you can run `pip install -r requirements.txt`.

**For NIQE:**

Out of scipy.misc.imresize deprecated, in NIQE.py we change
```python
img2 = scipy.misc.imresize(img, 0.5, interp='bicubic', mode='F')
```
to
```python
from PIL import Image
im = Image.fromarray(img)
size = tuple((np.array(im.size) * 0.5).astype(int))
img2 = np.array(im.resize(size, Image.BICUBIC))
```
According to [this](https://github.com/scipy/scipy/issues/6417), we think that `mode='F'` should keep the image original.

---

<center><strong>CLASS</strong> Metrics(mode, file_path, name="Metrics", use_gpu=True, A_name='_real_A', B_name='_fake_B', data_path='', A_path='', B_path='')</center>

To run it, you should provide two parameters: `mode` and `file_path`.

- `data_path`: the path to your data folder.

- `file_path`: the folder path you want to set scores. The scores will be stored in `.txt` file.

- `use_gpu`: **The default value is True when cuda is available**, and of course it will be False when cuda isn't available. If you don't want to use gpu, then you can set it False.

- `mode`: **You must set the mode value.** 

  '**mixed**' mode means that your first dataset and your second dataset are mixed but all of first dataset are named by ྾྾྾ and the other are named by ྾྾.By using this mode, you should set the parameter `data_path`.

  '**same_dir_parted**' mode means that your tow datasets are stored in two folders and the two folders are storted in the same father folder. And then you should set the parameter `data_path`. To emphasize it, its kid folders should be sorted alphabetically. The former will be the groundtruth dataset and the latter will be the low-light dataset. 

  '**diff_dir_parted**' mode means that your two datasets are stored in two folders and the two folders are storted in the different folders. So you should set the two parameters: `A_path`, `B_path`. And of course the former will be the groundtruth dataset and the latter will be the low-light dataset. 

- `A_name`, `B_name`: Only when mode is set as 'mixed', both are effective. The default value of `A_name` is `'_real_A'` and another is `'_fake_B'`. Those images of your groundtruth dataset are named by `྾྾྾_real_A` and those images of your low-light dataset are named by `྾྾྾_fake_B`.

- `A_path`, `B_path`: Only when mode is set as 'diff_dir_parted', both are effective. The former will be the path to the groundtruth dataset and the latter will be the path to the low-light dataset. 

---

Warning: For those no-reference metics, they will only load second datasets!

---

Now we will show the usage:

```python
import Metrics
metrics = Metrics(mode='mixed', data_path='/home/joy/ParaPata/', file_path='/home/joy/ParaPata/Pata/', A_name='_real', B_name='_fake')
MAE = metrics['MAE']
print(MAE)
```

Specially, you can use `metrics['All']` to record all metrics! 

To use the project, you should download these models following by the [link](https://drive.google.com/drive/folders/1EWk03SfEwVHf--7G-lr6P_o6EFuAUL3G?usp=sharing) and put them in `./models/`

If you need to reload datasets, you can call the object to reset some parameters. Here is the example:

```python
import Metrics
metrics = Metrics(mode='mixed', data_path='/home/joy/ParaPata/', file_path='/home/joy/ParaPata/Pata/')
MAE = metrics['MAE']
print(MAE)
metrics(mode='same_dir_parted', file_path='/home/joy/ParaPata/Pata/Olah', data_path='/home/joy/ParaPata/yahoo/')
MSE = metrics['']
```

