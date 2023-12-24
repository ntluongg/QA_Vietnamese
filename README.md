# User instruction
## Some note:
First you need to use Kaggle notebook with GPU T4 x2 enabled because this model is "data-parallel"ed.
And you also need to add the BKAI-IGH NeoPolyp dataset to your kaggle notebook (https://www.kaggle.com/competitions/bkai-igh-neopolyp/overview).
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