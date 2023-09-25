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
python3 organize_photos.py --org_cfg <LOCAL_PATH_CFG_YAML>
```
## Config Template
- use template config at [sample-files/template_config.yaml](sample-files/template_config.yaml)

## To run a quick test
- refer to sample config [here](sample-files/config.yaml). Make a directory named sample-input-images inside the root directory of this repo and upload a few sample images
- Run `python3 organize_photos.py --org_cfg sample-files/config.yaml`


# What can you do with this repo right now?
- organize photos by date
