# GST FP
Code for gst_fp experiment

## Dependencies
+ python3  
+ numpy  
+ pandas  
+ matplotlib  
+ sklearn  
+ cma  

Use `pip install -r requirements.txt` to install the dependencies you need.

## How to use
Now we support two materials: glass & GST, improvements are coming soon :D  
Here is an example to show how to use this program.  
First, type `python process.py` in cmd or terminal, and then input the information required:  
```
The material to test: 1.glass 2.GST 3.GeTe 4.AIST 5.others 3
Input the path to data file: ../0109/GeTe 20nm 0.csv
Using data: ../0109/GeTe 20nm 0.csv
Whether to use transfer: [Y/n] 
Transfer mode: ON
20001 data point(s) found
Choose the data type: 1.AM 2.CR (Now work on 1500-1600nm only) 1
GST phase: AM
Input the thickness(nm) of gst layer: (Now support 20nm and 80nm only) 20
Initial data error: 19820.4382976
Start the regression process? [y/N] y

```

