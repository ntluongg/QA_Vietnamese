# User instruction
## Some note:
First you need to use Kaggle notebook with GPU enabled.
And you also need to add our custom dataset https://www.kaggle.com/datasets/imnotluong/squadv2.
## guide:
```python
!pip install gdown
```
```python
import gdown

url = 'https://drive.google.com/drive/folders/19kntGT7S5kAU2yGfC3duCesg0F59ic52?usp=drive_link'
gdown.download_folder(url, quiet=True, use_cookies = False,)
```
```python
!git clone https://github.com/ntluongg/QA_Vietnamese.git
```
```python
!mkdir predicted
```
```python
!python /kaggle/working/QA_Vietnamese/infer.py --checkpoint '/kaggle/working/pretrained_bertQA_mid' --test_dir '/kaggle/input/squadv2/test_pair_without_punc' --out_dir '/kaggle/working/predicted'

# parse args checkpoint, test_dir (please add data in the link provided), out_dir