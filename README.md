# NeuACF
This is an implementation of paper  [(Aspect-Level Deep Collaborative Filtering via Heterogeneous Information Networks)](http://www.shichuan.org/doc/46.pdf). 


### Requirements

- Python 3.6
- Tensorflow 1.4

### How to Run

```
python ./acf.py ../dataset/amazon/ --mat "U.UIU,I.IUI,U.UITIU,I.ITI,U.UIVIU,I.IVI,U.UICIU,I.ICI" --epochs 40 --last_layer_size 64 --batch_size 1024 --num_of_neg 10 --learn_rate 0.00005 --num_of_layers 2 --mat_select median

```

### Parameters

Parameter | Note  
|:---|:---|
|--mat|sim_mat [default: ""]|
|--epochs|Embedding size [default: 40]|
|--last_layer_size| The number of iterations [default: 64]|
|--num_of_layers|                The number of layers [default: 2]|
|--num_of_neg|               The number of negs [default: 2]|
|--learn_rate|                The learn_rate [default: 0.00005]|
|--batch_size|                batch_size [default: 1024]|
|--mat_select|                mat select type [default: median]|
|--merge|                batch_size [default: attention]|


#### Link
For more information, visit the webpage http://www.shichuan.org