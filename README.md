# DeepIdentifier

This repository contains the pytorch implementation of **DeepIdentifier** as describe in following paper:
```
DeepIdentifier: A Deep Learning-Based Lightweight Approach for User Identity Recognition
```
Link:
```
https://link.springer.com/chapter/10.1007/978-3-030-35231-8_28
```

## Data
The UIR dataset is public to facilitate the research community and free to download. The one in folder "UIR_dataset" is the raw data we collected from smartphones, and the one in folder "prep_data" is the preprocessed data to fit the requirement of deep learning model input, which is used to perform the example.

UIR dataset is a realistic dataset we collected from ten subject swith seven activities: running, walking, going up stairs, going down stairs, jumping, rope jumping, and cycling. Each subject performs each activity with the same time duration, so the data is balance on both subjects and activities. Unlike those data collected in highly restricted environments by laboratory-quality devices, we adopted commercial mobile phone productsfor collection. We set the sampling rate to 100 Hz. To restore the real-world scenario, we performed multiple activities in one record from 2-min to 7-min. For example, we walked for 2-min and then ran for 3-min on the playground, and continued this cycle. The dataset contains more than a total of 30 hours of sensor data, which is much longer than that provided by any other data.

## Usage

### Requirements
- Python (**>=3.6**)
- PyTorch (**>=1.1.0**)
- scikit-learn

### Input
The preprocessed windows in the shape of (n, 1, 500, 6)

### Output
Classification result

## Citation
```
@inproceedings{lee2019deepidentifier,
  title={DeepIdentifier: A Deep Learning-Based Lightweight Approach for User Identity Recognition},
  author={Lee, Meng-Chieh and Huang, Yu and Ying, Josh Jia-Ching and Chen, Chien and Tseng, Vincent S},
  booktitle={Proceedings of the 15th International Conference on Advanced Data Mining and Applications},
  pages={389--405},
  year={2019},
  organization={Springer}
}
```
