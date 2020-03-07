# Audio classification using CNN and LSTM

<!-- Foobar is a Python library for dealing with word pluralization. -->

## Data Visualization

### MFCC Features

![mfcc](images/mfcc.png)

### Spectrogram
![mfcc](images/spec.png)
### Raw Audio

![mfcc](images/raw.png)
## Results



|   |Training   |Validation  |testing   |
|---|-----------|--------|--------------|
|Dice Loss      |0.084   |0.098   |0.098|
|Mean IOU       |0.776   |0.763   |0.762|


### Training Accuracy Plot
![accuracy_plot](images/Section-0-Panel-0-3919lbfls.png)


### Validation Accuracy Plot

![validation_accuracy](images/Section-0-Panel-1-jmn9k1muq.png)



### Training Error Plot
![error_plot](images/Section-0-Panel-2-99q67v5kr.png)


### Validation Error Plot

![validation_error](images/Section-0-Panel-3-tg061apfa.png)





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

```bash
python codes/baseline/main.py
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)