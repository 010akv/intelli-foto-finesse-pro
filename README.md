# foto-finesse-pro
A fun, AI-based utilities app for your precious memories. Elevate your photo finesse with cutting-edge organization, object detection, image classification and much more. 

# WIP
- this is a work in progress, stay tuned for more features!

# Installation
If you do not have pip installed, run `python setup-utils/get-pip.py`
```
conda create -n foto-finesse-pro
conda activate foto-finesse-pro
pip install -r requirements.txt
```

# Supported Image Formats
- png, jpg, jpeg

# Usage
## Organize Photos
```
python3 organize.py --config_file <LOCAL_PATH_CFG_YAML>
```

## Classify Photos
```
python3 classify.py --config_file <LOCAL_PATH_CFG_YAML>
```

## Config Template
- use template config at [config/template_config.yaml](config/template_config.yaml)

## To run a quick test
- refer to sample config [here](config/config.yaml). Make a directory named sample-input-images inside the root directory of this repo and upload a few sample images
- Organize: Run `python3 organize_photos.py --config_file config/config.yaml`
- Classify: Run `python3 classify_images.py --config_file config/config.yaml`


# What can you do with this repo right now?
- organize photos by date
- classify a given image into one of [1000 ImageNet Classes](https://deeplearning.cms.waikato.ac.nz/user-guide/class-maps/IMAGENET/)