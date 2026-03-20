# CNN-based-EEG-artifact-removal
An algorithim using CNN to denoise EEG data. Using z-score normalization to compress the EEg data to the rannge of 0-255(gray-scale map), then using U-net to train and denoise.

**Note**:This method will result in some loss of some time-domain information of the EEG data. Before choosing this model, it is necessary to consider whether one can accept such information loss.


## Training Order
Preprocessing -> Dataset -> U_net_model(No need to run) -> Train -> Predict -> Evaluate

**If there are some places can be imporved, then text me, thank you!**
