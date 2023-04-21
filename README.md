# MSCovidPaperCode

This repository contains code for the paper: 
Chikersal, P., Venkatesh, S., Masown, K., Walker, E., Quraishi, D., Dey, A., Goel, M., & Xia, Z. Predicting Multiple Sclerosis Outcomes during the COVID-19 Stay-at-Home Period: Observational Study Using Passively Sensed Behaviors and Digital Phenotyping. In JMIR mental health (2022).

https://prernaa.github.io/files/papers/chikersal2022_jmir_ms.pdf

**Please cite this paper if you use this code in your work.**

Note - the code does not contain the data used for the analysis. If you want access to the data, please email me at pchikersal@gmail.com. We will need to work out a data usage agreement to satisfy Carnegie Mellon University and University of Pittsburgh's IRB requirements. 

## Description of Files

- Feature Extraction
-- "8.Feature Extraction For Multimodal Sensor Pipeline (covid related, weekly).ipynb": runs the code for feature extraction from all sensors. the "passivefeatureslite" folder contains the feature extraction helper functions. 
- tochiBasedPipeline.py implements the ML pipeline for 1-sensor models. 
- tochiBasedPipelineAutomate.ipynb automates the process of tuning ML parameters for 1-sensor models. It calls tochiBasedPipeline.py via the command line to do so. 
-  
