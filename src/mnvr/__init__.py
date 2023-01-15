from .io.yaml_read import MnvrConfig

try:
    MnvrConfig()
except:
    print("There was a problem reading your mnvr.yaml file, you might have trouble using the package until this is resolved..")