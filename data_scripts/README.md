## Dataset Pre-processing
This folder contains code that will help you pre-process and save the [extened Cohn-Kanade (CK+)][CK+] and 
the Toronto Face Database (TFD) in ``.npy`` files to be used by our CNN models. 

> :pushpin: **Note:** We assume that you have already downloaded copies of the CK+ and TFD datasets 
on your machine. This code does not download the datasets from the distributors. It only pre-processes 
an already downloaded copy.

## Extended Cohn-Kanade Dataset (CK+)



## Toronto Face Database (TFD)

Once you have downloaded the TFD dataset, you should have a folder containing the following ``.mat`` files:

```shell
TFD_48x48.mat
TFD_96x96.mat
TFD_info.mat
```

The ``make_tfd_dataset.py`` file will help you convert the ``.mat`` files in the Toronto Face Dataset (TFD)
to ``.npy`` files. The script requires two arguments. They are:

1. **Input path:** path of folder that contains the ``.mat`` files
2. **Output/save path:** path to save the ``.npy`` files (default: ``./TFD_HERE/``)

To run the script, simply type the following:
```python
python make_tfd_dataset.py --input_path /path/to/mat/files --save_path /path/to/save/npy/files
```

For example, suppose the ``.mat`` files were stored in ``/data/TFD/`` and you wanted to save 
the ```.npy``` files to ``./TFD_HERE/``, then you would use the following command:
```python
python make_tfd_dataset.py --input_path /data/TFD/ --save_path ./TFD_HERE/
```

The ``TFD_HERE`` folder should now contain the following folders:
```shell
npy_files
  TFD_48
    split_0
    split_1
    split_2
    split_3
    split_4
    unlabeled
  TFD_96
    split_0
    split_1
    split_2
    split_3
    split_4
    unlabeled
```

Now, all you need to do is open the ``data_paths.py`` files in the ``tfd`` directory of this respository, 
and set it to the absolute path of ``TFD_HERE``.

Congratulations! You are now ready to run our CNNs on the TFD dataset!

[CK+]:http://www.pitt.edu/~emotion/ck-spread.htm
