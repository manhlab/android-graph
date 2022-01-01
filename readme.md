Android analys with functions call
```
pip3 install -e git+https://github.com/androguard/androguard.git#egg=androguard
pip install --pre dgl-cu110 -f https://data.dgl.ai/wheels-test/repo.html
pip install pyyaml==5.4.1
python /content/android-graph/core/process_dataset.py  --source-dir /content/dataset/Adware/Adware --dest-dir data2  
pip install torchtext

```

*** Note: