# Metrcis for Low Light Enhancement

Here we provide the project able to compute metrics for low light enhancement !

Metrics provided are these:

- MAE
- MSE
- PSNR
- SSIM
- LPIPS
- LOE
- NIQE
- SPAQ
- NIMA

To use it, you need to prepared these packages:

```yaml
scipy~=1.7.1
numpy~=1.19.5
torch~=1.9.0
lpips~=0.1.3
opencv-python~=4.5.3.56
torchvision~=0.10.0
pillow~=8.2.0
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

<center><strong>CLASS</strong> Metrics(data_path, file_path, use_gpu, mode, A_name, B_name)</center>

To run it, you should provide two parameters: data_path and file_path.

- data_path: the path to your data folder.

- file_path: the path you want to set scores.

- use_gpu: **The default value is True when cuda is available**, and of course it will be False when cuda isn't available. If you don't want to use gpu, then you can set it False.

- mode: **The default value is 'mixed'.** It means that your first dataset and your second dataset are mixed but all of first dataset are named by ྾྾྾ and the other are named by ྾྾.

  The parameter's another value is 'parted'. It means that your tow datasets are stored in two folders. And then there is no need to set `A_name` and `B_name` because they are ignored. To emphasize it, the folders should be sorted alphabetically. The former will be the first dataset and the latter will be the second dataset.

- `A_name`, `B_name`: Only when mode is set as 'mixed', both are effective. The default value of `A_name` is `'_real_A'` and another is `'_fake_B'`. Those images of your first dataset are named by `྾྾྾_real_A` and those images of your second dataset are named by `྾྾྾_fake_B`.

<center><font size="7" color="red">Warning: For those no-reference metics, they will load second datasets! </font></center>

---

Now we will show the usage:

```python
import Metrics
metrics = Metrics(data_path='/home/joy/ParaPata/', file_path='/home/joy/ParaPata/Pata/')
MAE = metrics['MAE']
print(MAE)
```

Specially, you can use `metrics['All']` to record all metrics!
