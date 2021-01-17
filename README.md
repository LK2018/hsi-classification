Description
-----------------
This is a project that implemented several published methods of HSI 
classification. Users can set parameters by editing `./common/config.yaml`, and 
users need to modified the data directories by editing `./common.get_config.py`.
Users run the code by:

```
python main.py
```

Users can change `main.py` according to their own needs.

Requirements
------------
* python3.7  
* pytorch  
* keras
* tensorflow  
* yacs
* deep-belief-network (https://github.com/albertbup/deep-belief-network.git) 
* bayesian-optimization (pip install bayesian-optimization) 