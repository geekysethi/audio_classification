# Audio classification using CNN and LSTM

<!-- Foobar is a Python library for dealing with word pluralization. -->

## Data Visualization

### MFCC Features
<img src="images/mfcc.png" width="400">

### Spectrogram
<img src="images/spec.png" width="400">


### Raw Audio

<img src="images/raw.png" width="400">

## Results



|   |Training   |Validation  |testing   |
|---|-----------|--------|--------------|
|Dice Loss      |0.084   |0.098   |0.098|
|Mean IOU       |0.776   |0.763   |0.762|

### CNN

|:------------:|:-----------:|:----------:|:-----:|:----------:|
|    Dataset   | Spectrogram |            |  MFCC |            |
|              |    Train    | Validation | Train | Validation |
| urbansound8k |             |            |       |            |

### Training Accuracy Plot

<img src="images/Section-0-Panel-0-3919lbfls.png" width="500">


### Validation Accuracy Plot
<img src="images/Section-0-Panel-1-jmn9k1muq.png" width="500">





### Training Error Plot
<img src="images/Section-0-Panel-2-99q67v5kr.png" width="500">




### Validation Error Plot

<img src="images/Section-0-Panel-3-tg061apfa.png" width="500">





## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install foobar.

```bash
pip install requirements.txt 
```

or

```bash
conda create --name <env> --file requirements.txt 
```

## Usage
### Dataset 
[urbansound8k]

### Pre-process Data

```bash
python codes/pre_processing/pre_processing_urbansound.py
```
### Train and Test
```bash
python codes/baseline/main.py
```


[urbansound8k]: [https://urbansounddataset.weebly.com/urbansound8k.html]

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.


## License
[MIT](https://choosealicense.com/licenses/mit/)