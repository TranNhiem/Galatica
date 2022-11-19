
import tqdm
import urllib
import os 

model_path="/data1/pretrained_weight/Glactica30b/galactica-30b"
#mo/home/harry/BLIRL/SSL_Applications/Dreamboothdel_path="/data1/pretrained_weight/Glatica/galactica-30b"
#model_path="/data1/pretrained_weight/Glatica/galactica-6.7b"
TOKENIZER_URL = 'https://dl.fbaipublicfiles.com/galactica/tokenizer.json'
WEIGHT_DIR = 'https://dl.fbaipublicfiles.com/galactica'

MINI_FILES = [
    'config.json',
    'pytorch_model.bin'
]

BASE_FILES = [
    'config.json',
    'pytorch_model.bin'
]

STANDARD_FILES = [
    'config.json',
    'pytorch_model-00001-of-00002.bin',
    'pytorch_model-00002-of-00002.bin',
    'pytorch_model.bin.index.json'
]

LARGE_FILES = [
    'config.json',
    'pytorch_model-00001-of-00007.bin',
    'pytorch_model-00002-of-00007.bin',
    'pytorch_model-00003-of-00007.bin',
    'pytorch_model-00004-of-00007.bin',
    'pytorch_model-00005-of-00007.bin',
    'pytorch_model-00006-of-00007.bin',
    'pytorch_model-00007-of-00007.bin',
    'pytorch_model.bin.index.json'
]

HUGE_FILES = [
    'config.json',
    'pytorch_model-00001-of-00026.bin',
    'pytorch_model-00002-of-00026.bin',
    'pytorch_model-00003-of-00026.bin',
    'pytorch_model-00004-of-00026.bin',
    'pytorch_model-00005-of-00026.bin',
    'pytorch_model-00006-of-00026.bin',
    'pytorch_model-00007-of-00026.bin',
    'pytorch_model-00008-of-00026.bin',
    'pytorch_model-00009-of-00026.bin',
    'pytorch_model-00010-of-00026.bin',
    'pytorch_model-00011-of-00026.bin',
    'pytorch_model-00012-of-00026.bin',
    'pytorch_model-00013-of-00026.bin',
    'pytorch_model-00014-of-00026.bin',
    'pytorch_model-00015-of-00026.bin',
    'pytorch_model-00016-of-00026.bin',
    'pytorch_model-00017-of-00026.bin',
    'pytorch_model-00018-of-00026.bin',
    'pytorch_model-00019-of-00026.bin',
    'pytorch_model-00020-of-00026.bin',
    'pytorch_model-00021-of-00026.bin',
    'pytorch_model-00022-of-00026.bin',
    'pytorch_model-00023-of-00026.bin',
    'pytorch_model-00024-of-00026.bin',
    'pytorch_model-00025-of-00026.bin',
    'pytorch_model-00026-of-00026.bin',
    'pytorch_model.bin.index.json'
]

CHECKPOINT_PATHS = {
 
    'mini': [WEIGHT_DIR + '/125m/' + file for file in MINI_FILES],
    'base': [WEIGHT_DIR + '/1.3b/' + file for file in BASE_FILES],
    'standard': [WEIGHT_DIR + '/6.7b/' + file for file in STANDARD_FILES],
    'large': [WEIGHT_DIR + '/30b/' + file for file in LARGE_FILES],
    'huge': [WEIGHT_DIR + '/120b/' + file for file in HUGE_FILES]
   
}


class DownloadProgressBar(tqdm.tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def _download_file(file_url: str, file_loc: str):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=file_url.split('/')[-1]) as t:
        urllib.request.urlretrieve(file_url, filename=file_loc, reporthook=t.update_to)

def download_tokenizer(tokenizer_path: str):
    _download_file(TOKENIZER_URL, tokenizer_path)


def download_model(model_name: str, model_path: str):

    for file_url in tqdm.tqdm(CHECKPOINT_PATHS[model_name]):
        file_loc = os.path.join(model_path, file_url.split('/')[-1])
        if os.path.exists(file_loc):
            continue
        _download_file(file_url, file_loc)


download_model("large",model_path)
download_tokenizer(model_path,)