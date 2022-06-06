```python
import os  # Import os to handle folders and files
import time  # Import module to check code time executing
import random  # Import module to do pseudo-random numbers
import numpy as np  # Import numpy module to read tables, manage data, etc
from scripts import main  # Import main script
from prettytable import PrettyTable  # Import module to do tables
```

# T-distributed Stochastic Neighbor Embedding (t-SNE) in Swift Data
## Introduction GRBs
Gamma-ray bursts are the brightest events in the universe. Regardind their duration, GRBs shows a binomial distribution with a cut off around 2 seconds. At the beginning, this suggested the presence of two kind of populations, and therefore, two kind of progenitors. Nevertheless, this classifications is very ambiguous. 


#Previous works in Machine Learning of Classification
As suggested by [Jespersen et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020ApJ...896L..20J/abstract), Swift GRBs can be separated into two groups when t-SNE is performed. In this Jupyter notebook, we replicate this work by adding more recent data and an in-depth analysis of t-SNE performance. Moreover, we hope to add synthetic GRB performed using Machine Learning instances and join into Swift and other GRB data packages.

In this document, we create two scripts named `main.py` and `helpers.py`. The first one does the main things about implementing tSNE and preparing the data, whereas the second one only do some repetitive tasks and minor jobs. 
Note: the `main.py` script needs to download some tables previously to work, doing this is so easy, you only need to use the `summary_tables_download` instance in `main.py`, but, before doing this, you need to create an object to initialize `SwiftGRBWorker` class:


```python
%matplotlib inline
object1 = main.SwiftGRBWorker()
# object1.summary_tables_download()  # Comment this line if you already have downloaded the tables
```

## Swift Data Download
Firstly, we get all the available data from [Swift Database](https://swift.gsfc.nasa.gov/results/batgrbcat/) to prepare them to perform t-SNE. For this, In the case were, we only wanted to download only data for one GRB, it is needed to use `download_data` instance from `main.py` class and pass it a name, TRIGGER_ID, and observation resolution, that we get from `summary_general` table:


```python
tables_path = object1.table_path
sum_general_path = os.path.join(tables_path, "summary_general.txt")
GRB_names, ids = np.genfromtxt(sum_general_path, delimiter="|", dtype=str, usecols=(0, 1), unpack=True, autostrip=True)
print(f"Names: {GRB_names}")
print(f"IDs: {ids}")
```

    Names: ['GRB200829A' 'GRB200819A' 'GRB200809B' ... 'GRB041219B' 'GRB041219A'
     'GRB041217']
    IDs: ['993768' '992099' '987745' ... '100368' '100307' '100116']
    

Then, we download $64ms$ data for a random GRB listed before:


```python
# result_GRB = [[], []]  # Comment this line if you already have downloaded the data and you need to customize any GRB
result_GRB = ['GRB060614', None]  # Put Any customized GRB here
while result_GRB[1] is not None:
    random_position = random.randint(0, len(GRB_names))
    result_GRB = object1.download_data(GRB_names[random_position], ids[random_position])
    print(f"{result_GRB[0]} has been downloaded") if not result_GRB[1] else print(f"Error downloading {GRB_names[random_position]} data")
```

Columns of this table refer to:
* Time since BAT Trigger time (s)
* Intensity and its error in bands 15-25keV, 25-50keV, 50-100keV, 100-350keV, 15-350keV in counts/sec/det


```python
x = PrettyTable()  # Create printable table
original_data_path = object1.original_data_path
data = np.genfromtxt(os.path.join(original_data_path, f"{result_GRB[0]}_{64}ms.gz"), autostrip=True)  # Read data
column_names = ('Time (s)', '15-25keV', 'Error', '25-50keV', 'Error', '50-100keV', 'Error', '100-350keV', 'Error','15-350keV', 'Error')
[x.add_column(column_names[i], np.round(data[:3, i], 3)) for i in range(len(data[0]))]  # Add initial 3 rows to each column
x.add_row(['...']*len(column_names))  # Add dots space
x.add_rows(np.round(data[-3:-1], decimals=3))  # Add last 2 rows to each column
print(x)
```

    +----------+----------+-------+----------+-------+-----------+-------+------------+-------+-----------+-------+
    | Time (s) | 15-25keV | Error | 25-50keV | Error | 50-100keV | Error | 100-350keV | Error | 15-350keV | Error |
    +----------+----------+-------+----------+-------+-----------+-------+------------+-------+-----------+-------+
    | -239.728 |  -0.032  | 0.052 |  -0.031  |  0.05 |   -0.016  | 0.045 |   -0.055   | 0.037 |   -0.135  | 0.093 |
    | -239.664 |  -0.048  | 0.057 |  0.075   | 0.048 |   -0.028  | 0.057 |   -0.017   |  0.04 |   -0.018  | 0.102 |
    |  -239.6  |  -0.069  | 0.048 |  0.011   | 0.048 |   0.041   | 0.046 |   -0.005   | 0.029 |   -0.022  | 0.087 |
    |   ...    |   ...    |  ...  |   ...    |  ...  |    ...    |  ...  |    ...     |  ...  |    ...    |  ...  |
    | 482.128  |   0.0    |  0.0  |   0.0    |  0.0  |    0.0    |  0.0  |    0.0     |  0.0  |    0.0    |  0.0  |
    | 482.192  |   0.0    |  0.0  |   0.0    |  0.0  |    0.0    |  0.0  |    0.0     |  0.0  |    0.0    |  0.0  |
    +----------+----------+-------+----------+-------+-----------+-------+------------+-------+-----------+-------+
    

Additionally, if you want to reproduce the original light curves for this GRB, you can use the `plot_any_grb` function using the argument _t_ in False (indicating that you don't need to filter the LC). In this case, it takes the form:


```python
%matplotlib inline
fig, axes = object1.plot_any_grb(result_GRB[0], t=False)  # Plot all extension of Light Curve
```


    
![png](README_files/README_10_0.png)
    


In order to download hundreds of data files (that is my case), it is faster to do Threading over all names and IDs that repeat the process one by one (e.g. `for` instance), so , I propose to use the `so_much_downloads` function:


```python
if None:  # Change None to any True variable in if statement if you need to download data (by first time using by example)
    time1 = time.perf_counter()
    object1.so_much_downloads(GRB_names, ids)
    time2 = time.perf_counter()
    GRB_errors = np.genfromtxt(os.path.join(original_data_path, f"Errors_{object1.res}.txt"), delimiter='|', dtype=str, unpack=True)[0]
    print(f"Downloaded {len(GRB_names)-len(GRB_errors)} light curves in {round(time2-time1, 2)}s (Total = {len(GRB_names)})")
```

This function can take a little time to run, but it is so much faster than a loop. Additionally, some GRBs could not be downloaded, you can check them using the `Errors.txt` file, we want to delete this GRBs for the total name list:


```python
GRB_errors = np.genfromtxt(os.path.join(original_data_path, f"Errors_{object1.res}.txt"), delimiter='|', dtype=str, unpack=True)[0]
print(f'{len(GRB_errors)} GRBs get download error: {GRB_errors}')
GRB_names = list(set(GRB_names) - set(GRB_errors))
Initial_GRB_names = GRB_names  # Array with all initial GRBs downloaded
```

    38 GRBs get a "download error": ['GRB170131A' 'GRB160623A' 'GRB160506A' 'GRB160409A' 'GRB151118A'
     'GRB150407A' 'GRB140909A' 'GRB140611A' 'GRB131031A' 'GRB130913A'
     'GRB130816B' 'GRB130604A' 'GRB130518A' 'GRB121226A' 'GRB120817B'
     'GRB120728A' 'GRB111103A' 'GRB111005A' 'GRB110604A' 'GRB101204A'
     'GRB100621A' 'GRB100213B' 'GRB090827' 'GRB090720A' 'GRB090712'
     'GRB081211A' 'GRB071112C' 'GRB071028B' 'GRB071010C' 'GRB071006'
     'GRB070227' 'GRB070125' 'GRB060123' 'GRB051221B' 'GRB050724' 'GRB050716'
     'GRB050714B' 'GRB041219A']
    

In summary, these errors could occur for two reasons:
* Firstly, the GRB doesn't have any Trigger ID
* Secondly, the GRB has Trigger ID, but no data.

We close this section by remarking that original size data can use 2.67GB of free space on disk approximately (in decompress mode). Nevertheless, compressing data using `gzip` library will save much room:


```python
size = 0  # Set size variable to zero
for path, dirs, files in os.walk(object1.original_data_path):  # Loop over the folder containing all data downloaded
    for f in files:  # Loop over files into folder
        fp = os.path.join(path, f)  # Join file name with folder path
        size += os.stat(fp).st_size  # Get file size and sum over previous size
print(f"There are {round(size/(1024*1024), 3)} MB of data")
```

    There are 895.911 MB of data
    

# Swift Data Pre-processing
In order to prepare data for tSNE implementation, we follow four steps:
* Limit all GRBs out of $T_{100}$
* Normalize light curves (lc) by total fluence in 15-350keV band
* Pad with zeros all GRBs, putting then in the same time standard basis
* Concatenate data from all bands and perform DFT to get Fourier amplitude spectrum

In the next sections, we are going to describe these processes and show how to use some functions to do these steps.

## First step: Limit lc out of $T_{100}$
We extract the durations for all GRBs available in the file `summary_burst_durations.txt` using the `durations_checker` instance. If you don't pass any GRB name, this function returns a list containing three values for each GRB in the table: Name, $T_{i}$ start and end times (in seconds), where $i$ can be 50, 90, or 100 (default value), but if you pass it a name, then it returns these values only by this GRB:


```python
durations_times = object1.durations_checker(result_GRB[0])
print(f"{durations_times[0, 0]} has T_100={round(float(durations_times[0, 2])-float(durations_times[0, 1]), 3)}s (T_100 start={durations_times[0, 1]}s, T_100 end={durations_times[0, 2]}s)")
```

    GRB060614 has T_100=180.576s (T_100 start=-1.496s, T_100 end=179.080s)
    

With these values, we can limit our GRB lc using the instance `lc_limiter`.  In this function, it is possible directly constrains on the time or pass it an integer to indicate what duration we need (however, these integers can be only 50, 90, and 100). So, we try to extract the lc out of $T_{100}$ by using an integer and setting limits manually, then comparing both:


```python
limited_data_1 = object1.lc_limiter(result_GRB[0])  # Limiting by T_100
limited_data_2 = object1.lc_limiter(result_GRB[0], limits=(durations_times[0, 1], durations_times[0, 2]))  # By values
print(f"Are both arrays equal? Answer={np.array_equal(limited_data_1, limited_data_2)}")
x.clear_rows()  # Clear rows of data
x.add_rows(np.round(limited_data_1[:3], decimals=3))  # Add first 3 rows to each column
x.add_row(['...']*len(column_names))  # Add dots space
x.add_rows(np.round(limited_data_1[-3:-1], decimals=3))  # Add last 2 rows to each column
print(x)
```

    Are both arrays equal? Answer=True
    +----------+----------+-------+----------+-------+-----------+-------+------------+-------+-----------+-------+
    | Time (s) | 15-25keV | Error | 25-50keV | Error | 50-100keV | Error | 100-350keV | Error | 15-350keV | Error |
    +----------+----------+-------+----------+-------+-----------+-------+------------+-------+-----------+-------+
    |  -1.456  |   0.12   | 0.077 |  0.243   |  0.07 |   0.177   | 0.061 |   0.059    | 0.052 |   0.599   | 0.131 |
    |  -1.392  |   0.2    | 0.077 |  0.125   | 0.091 |    0.28   |  0.07 |   -0.012   | 0.051 |   0.593   | 0.148 |
    |  -1.328  |  0.139   | 0.064 |  0.174   | 0.085 |   0.251   | 0.076 |   0.052    | 0.052 |   0.616   |  0.14 |
    |   ...    |   ...    |  ...  |   ...    |  ...  |    ...    |  ...  |    ...     |  ...  |    ...    |  ...  |
    | 178.896  |  -0.003  | 0.018 |  0.009   | 0.018 |   0.019   | 0.016 |    0.01    | 0.015 |   0.035   | 0.034 |
    |  178.96  |  0.012   | 0.019 |  -0.008  | 0.018 |   -0.027  | 0.015 |   0.017    | 0.014 |   -0.006  | 0.033 |
    +----------+----------+-------+----------+-------+-----------+-------+------------+-------+-----------+-------+
    

Note that both methods are equivalent, and the lc values are now between start and end times for $T_{100}$. Note that both methods are equivalent, and the lc values are now between start and end times for $T_{100}$. Graphically, the lc out of $T_{100}$ is:


```python
fig_limited, axes_limited = object1.plot_any_grb(result_GRB[0], t=100)  # Plot Light Curve out of T_100
```


    
![png](README_files/README_22_0.png)
    


So, the next step is to do this for all downloaded GRBs, to get a much faster performance of execution, we can use the `so_much_lc_limiters` function:


```python
y = PrettyTable()  # Create printable table
time1 = time.perf_counter()
limited_data, GRB_names, errors = object1.so_much_lc_limiters(GRB_names)
time2 = time.perf_counter()
print(f"{len(limited_data)} GRBs limited in {round(time2-time1, 2)}s ({len(errors)} Errors)")
column_names_2 = ('Name', 't_start', 't_end', 'Error Type')
[y.add_column(column_names_2[i], errors[:, i]) for i in range(len(errors[0]))]  # Add rows to each column
print(y)
```

    LC Limiting: 100%|██████████| 1351/1351 [04:07<00:00,  5.47GRB/s]
    

    1300 GRBs limited in 248.18s (51 Errors)
    +------------+---------+---------+------------+
    |    Name    | t_start |  t_end  | Error Type |
    +------------+---------+---------+------------+
    | GRB160601A |  -0.024 |   0.12  |  Length=2  |
    | GRB130305A |   0.0   |   57.6  | Only zeros |
    | GRB070923  |  -0.004 |   0.04  |  Length=1  |
    | GRB151107A |         |         | ValueError |
    | GRB150101B |   0.0   |  0.016  |  Length=1  |
    | GRB080315  |         |         | ValueError |
    | GRB170524A |   0.0   |   0.12  |  Length=2  |
    | GRB050202  |  0.004  |  0.132  |  Length=2  |
    | GRB070209  |   0.0   |  0.084  |  Length=2  |
    | GRB160525A |         |         | ValueError |
    | GRB160501A |         |         | ValueError |
    | GRB051105A |  -0.004 |  0.064  |  Length=2  |
    | GRB150710B |         |         | ValueError |
    | GRB081105A |   0.0   |   9.6   | Only zeros |
    | GRB200324A |         |         | ValueError |
    | GRB070810B |  -0.008 |  0.076  |  Length=1  |
    | GRB060510A |  -6.752 |  16.748 | Only zeros |
    | GRB101225A |         |         | ValueError |
    | GRB061027  |         |         | ValueError |
    | GRB060502B |  -0.004 |  0.172  |  Length=2  |
    | GRB140622A |  -0.024 |  0.128  |  Length=2  |
    | GRB170112A |  -0.004 |  0.056  |  Length=1  |
    | GRB060728  |         |         | ValueError |
    | GRB100206A |  -0.008 |  0.124  |  Length=2  |
    | GRB050509B |   0.0   |  0.028  |  Length=0  |
    | GRB170906B | -11.776 |  7.424  |  Length=0  |
    | GRB161104A |  -0.016 |   0.1   |  Length=2  |
    | GRB090417A |  -0.004 |   0.08  |  Length=1  |
    | GRB141102A |   0.0   |   14.4  | Only zeros |
    | GRB190326A |   0.0   |  0.104  |  Length=1  |
    | GRB101201A |   0.0   |   67.2  | Only zeros |
    | GRB070126  |         |         | ValueError |
    | GRB081017  |         |         | ValueError |
    | GRB120218A | -26.696 |  8.312  | Only zeros |
    | GRB041219B |         |         | ValueError |
    | GRB090515  |  0.008  |  0.056  |  Length=1  |
    | GRB120305A |   0.0   |  0.128  |  Length=2  |
    | GRB061218  |         |         | ValueError |
    | GRB140311B |         |         | ValueError |
    | GRB050925  |  -0.036 |  0.068  |  Length=2  |
    | GRB080822B |         |         | ValueError |
    | GRB060218  |         |         | ValueError |
    | GRB050906  |         |         | ValueError |
    | GRB150101A |   0.0   |  0.068  |  Length=1  |
    | GRB110420B |  -0.004 |   0.1   |  Length=1  |
    | GRB150424A |         |         | IndexError |
    | GRB130822A |  -0.004 |  0.044  |  Length=0  |
    | GRB100628A |  -0.004 |  0.036  |  Length=1  |
    | GRB180718A |   0.0   |   0.1   |  Length=1  |
    | GRB131105A |  6.792  | 125.336 | Only zeros |
    | GRB090621B |  -0.028 |  0.144  |  Length=2  |
    +------------+---------+---------+------------+
    

This function returns a tuple of limited arrays and errors for all GRBs. In this case, 51 GRBs have any of the following errors:
* _FileNotFoundError_ if GRB does not have any data file downloaded in the selected resolution.
* _ValueError_ if the code can't get any limit values.
* _IndexError_ if the GRB does not appear in the `summary_burst_durations` table.
* _Length={value}_ if the limited GRB data has less than three discrete points.
* _Only zeros_ if the limited GRB data only has zeros.

Additionally, the second argument returned by `so_much_lc_limiters` is a GRB names array indicating the order of results (the first argument returned). This order is now our `GRB_names` variable because it does not have the error names.

To verify that the `so_much_lc_limiters` instance is working properly, we compare (for the random GRB selected before) if the data stored in _limited_data_1_ and obtained in parallelizing are equal:


```python
random_index = GRB_names.index(result_GRB[0])  # Search random GRB in new GRB names list
limited_data_random_GRB = np.array(limited_data[random_index])  # Extract limited values for random GRB
print(f"Are both arrays equal? Answer={np.array_equal(limited_data_1, limited_data_random_GRB)}")
```

    Are both arrays equal? Answer=True
    

## Second step: Normalizing Light Curves

In order to compare only the shape and behaviour of the light curves and not the fluences, we normalize our lc by total time-integrated flux, using the routine `lc_normalizer` which is based on the Simpson's Rule. This function returns a tuple normalized data, total time-integrated flux if _print_area_ parameter is _True_, otherwise only returns normalized data:


```python
normalized_data_random_GRB, total_flux = object1.lc_normalizer(limited_data_random_GRB, print_area=True)
x.clear_rows()  # Clear rows of data
x.add_rows(np.round(normalized_data_random_GRB[:2], decimals=4))  # Add new rows of limited data
x.add_row(['...'] * len(column_names))  # Add dots space
x.add_rows(np.round(normalized_data_random_GRB[np.r_[-2:0]], decimals=4))  # Add new rows of limited data
print(f"Total time-integrated flux for {result_GRB[0]} is: {round(total_flux, 4)}")
print(x)
```

    Total time-integrated flux for GRB060614 is: 38.6704
    +----------+----------+--------+----------+--------+-----------+--------+------------+--------+-----------+--------+
    | Time (s) | 15-25keV | Error  | 25-50keV | Error  | 50-100keV | Error  | 100-350keV | Error  | 15-350keV | Error  |
    +----------+----------+--------+----------+--------+-----------+--------+------------+--------+-----------+--------+
    |  -1.456  |  0.0031  | 0.002  |  0.0063  | 0.0018 |   0.0046  | 0.0016 |   0.0015   | 0.0013 |   0.0155  | 0.0034 |
    |  -1.392  |  0.0052  | 0.002  |  0.0032  | 0.0024 |   0.0072  | 0.0018 |  -0.0003   | 0.0013 |   0.0153  | 0.0038 |
    |   ...    |   ...    |  ...   |   ...    |  ...   |    ...    |  ...   |    ...     |  ...   |    ...    |  ...   |
    |  178.96  |  0.0003  | 0.0005 | -0.0002  | 0.0005 |  -0.0007  | 0.0004 |   0.0004   | 0.0004 |  -0.0002  | 0.0009 |
    | 179.024  | -0.0002  | 0.0005 | -0.0002  | 0.0005 |   0.0002  | 0.0004 |  -0.0001   | 0.0004 |  -0.0002  | 0.0009 |
    +----------+----------+--------+----------+--------+-----------+--------+------------+--------+-----------+--------+
    

#Note that normalized data are only limited data divided by 15-350keV integrated flux. So, the next step is to do this for all limited GRBs, to get a much faster performance of execution, we can use the `so_much_normalize` function. To check if this instance is doing its job well, we are going to compare (for the random GRB selected before) if the data stored in _normalized_data_random_GRB_ and obtained in parallelizing are equal:

We repeat for every GRB using the `so_much_normalize` function. We verify this step by comparin (for the random GRB selected before) if the data stored in _normalized_data_random_GRB_ and obtained in parallelizing are equal

```python
normalized_data = object1.so_much_normalize(limited_data)  # Normalizing all light curves
normalized_data_random_GRB_2 = normalized_data[random_index]  # Extract normalized values for random GRB
print(f"Are both arrays equal? Answer={np.array_equal(normalized_data_random_GRB, normalized_data_random_GRB_2)}")
```

    LC Normalizing: 100%|██████████| 1300/1300 [00:11<00:00, 108.92GRB/s]
    

    Are both arrays equal? Answer=True
    

## Third step: Zero Padding
With all GRBs limited out of $T_{100}$ and normalized, we need now to zero-pad their light curves to place them on the same time basis. The `zero_pad` instance performs this job by checking the max length of a data set and looking for the best suitable array size to do Fast Fourier Transform (FFT, the next step in data pre-processing).

Here, we are going to see how this function zero pad the data at its end for the random selected GRB before:


```python
zero_padded_data = object1.so_much_zero_pad(normalized_data)
zero_padded_data_random_GRB = zero_padded_data[random_index]
print(f"Best FFT suitable data length: {len(zero_padded_data_random_GRB)}")
x.clear_rows()  # Clear rows of data
x.add_rows(np.round(zero_padded_data_random_GRB[:2], decimals=4))  # Add new rows of limited data
x.add_row(['...'] * len(column_names))  # Add dots space
x.add_rows(np.round(zero_padded_data_random_GRB[len(normalized_data_random_GRB)-2:len(normalized_data_random_GRB)+2], decimals=4))  # Add new end rows of limited data
x.add_row(['...'] * len(column_names))  # Add dots space
x.add_rows(np.round(zero_padded_data_random_GRB[-3:-1], decimals=4))  # Add new end rows of limited data
print(x)
```

    LC Zero-Padding: 100%|██████████| 1300/1300 [00:20<00:00, 63.03GRB/s] 
    

    Best FFT suitable data length: 15309
    +----------+----------+--------+----------+--------+-----------+--------+------------+--------+-----------+--------+
    | Time (s) | 15-25keV | Error  | 25-50keV | Error  | 50-100keV | Error  | 100-350keV | Error  | 15-350keV | Error  |
    +----------+----------+--------+----------+--------+-----------+--------+------------+--------+-----------+--------+
    |  -1.456  |  0.0031  | 0.002  |  0.0063  | 0.0018 |   0.0046  | 0.0016 |   0.0015   | 0.0013 |   0.0155  | 0.0034 |
    |  -1.392  |  0.0052  | 0.002  |  0.0032  | 0.0024 |   0.0072  | 0.0018 |  -0.0003   | 0.0013 |   0.0153  | 0.0038 |
    |   ...    |   ...    |  ...   |   ...    |  ...   |    ...    |  ...   |    ...     |  ...   |    ...    |  ...   |
    |  178.96  |  0.0003  | 0.0005 | -0.0002  | 0.0005 |  -0.0007  | 0.0004 |   0.0004   | 0.0004 |  -0.0002  | 0.0009 |
    | 179.024  | -0.0002  | 0.0005 | -0.0002  | 0.0005 |   0.0002  | 0.0004 |  -0.0001   | 0.0004 |  -0.0002  | 0.0009 |
    | 179.088  |   0.0    |  0.0   |   0.0    |  0.0   |    0.0    |  0.0   |    0.0     |  0.0   |    0.0    |  0.0   |
    | 179.152  |   0.0    |  0.0   |   0.0    |  0.0   |    0.0    |  0.0   |    0.0     |  0.0   |    0.0    |  0.0   |
    |   ...    |   ...    |  ...   |   ...    |  ...   |    ...    |  ...   |    ...     |  ...   |    ...    |  ...   |
    | 978.128  |   0.0    |  0.0   |   0.0    |  0.0   |    0.0    |  0.0   |    0.0     |  0.0   |    0.0    |  0.0   |
    | 978.192  |   0.0    |  0.0   |   0.0    |  0.0   |    0.0    |  0.0   |    0.0     |  0.0   |    0.0    |  0.0   |
    +----------+----------+--------+----------+--------+-----------+--------+------------+--------+-----------+--------+
    

## Final Step: Discrete Fourier Transform
Finally, the last step of Swift data pre-processing is to perform a Fast Fourier Transform to zero-padded normalized data out of $T_{100}$. There are so many python packages to do this job, particularly in this notebook, we are going to use _scipy_ in the `fourier_concatenate` instance, but before that, this function concatenate all energy band measurements in one single array, as required to execute DFT:


```python
dft_random_GRB, dft_fig = object1.fourier_concatenate(zero_padded_data[random_index], plot=True, name=result_GRB[0])
```


    
![png](README_files/README_34_0.png)
    


Note that DFT data is below the Nyquist frequency, following the Nyquist-Shannon sampling theorem:

_The Nyquist-Shannon sampling theorem states that a signal sampled at a rate can be fully reconstructed if it contains only frequency components below half that sampling frequency. Thus the highest frequency output from the DFT is half the sampling rate._

With this, we can now calculate DFT for the entire zero-padded dataset using the `so_much_fourier` instance:


```python
pre_processing_data = object1.so_much_fourier(zero_padded_data)
```

    Performing DFT: 100%|██████████| 1300/1300 [00:20<00:00, 63.65GRB/s] 
    

Finally, the pre-processing data stage is over. Then, we want to save all data in a compressed format to load in the next section. For this, you can use the `save_data` function (based in `savez_compressed` instance of Numpy):


```python
object1.save_data(f"DFT_Preprocessed_data_{object1.res}ms", names=GRB_names, data=pre_processing_data)
```

# t-SNE in Swift Data
t-Distributed Stochastic Neighbor Embedding (or t-SNE) is a popular non-linear dimensionality reduction technique used for visualizing high dimensional data sets. After pre-processing Swift data in the $x_i$ vectors with Fourier Amplitudes, we want to perform this method by taking so much care when we read the results. Why? The t-SNE algorithm doesn’t always produce similar output on successive runs, and it depends on some hyperparameters related to the optimization process.

In this study, the most relevant hyperparameters on the cost function are (following the scikit-Learn and open-TSNE packages documentation):
* __Perplexity__: The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. Larger datasets usually require a larger perplexity. Note that perplexity linearly impacts runtime i.e. higher values of perplexity will incur longer execution time.
* __learning_rate__: The learning rate controls the step size of the gradient updates. If the learning rate is too high, the data may look like a ‘ball’ with any point approximately equidistant from its nearest neighbours. If the learning rate is too low, most points may look compressed in a dense cloud with few outliers.
* __metric__: The metric to use when calculating distance between instances in a feature array.

## t-SNE convergency
First of all, we want to see how t-SNE converges in the pre-processed data. To do this, we use the `convergence_animation` function, it is based in [tsne_animate](https://github.com/sophronesis/tsne_animate) package from GitHub in its `tsne_animation` function. But, before we need to load the pre-processing data saved:


```python
data_loaded = np.load(os.path.join(object1.results_path, f"DFT_Preprocessed_data_{object1.res}ms.npz"))
GRB_names, features = data_loaded['GRB_Names'], data_loaded['Data']
print(f"There are {len(GRB_names)} GRBs loaded: {GRB_names}")
```

    There are 1300 GRBs loaded: ['GRB140930B' 'GRB080702B' 'GRB150323B' ... 'GRB051109B' 'GRB060512'
     'GRB181027A']
    

Now, we will index GRBs durations (using the `durations_checker` instance) to see the results dependence with this feature:


```python
durations_data_array = object1.durations_checker(GRB_names, t=90)  # Check for name, t_start, and t_end
start_times, end_times = durations_data_array[:, :, 1].astype(float), durations_data_array[:, :, 2].astype(float)
durations = np.reshape(end_times - start_times, len(durations_data_array))  # T_90 is equal to t_end - t_start
```

    Finding Durations: 100%|██████████| 1300/1300 [00:02<00:00, 540.51GRB/s] 
    

Then we set the standard _perplexity_ value (30) from [Jespersen et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020ApJ...896L..20J/abstract), set auto _learning rate_ in scikit-Learn t-SNE implementation, and perform the animation:


```python
file_name = os.path.join('README_files', 'convergence_animation_pp_30.gif')
object1.convergence_animation(features, filename=file_name, perplexity=30, duration_s=durations)
```

![](README_files/convergence_animation_pp_30.gif)

As you can see, there is a clear dependence on $T_{90}$ duration and GRB position in the final plot (except for some GRBs, i. e. GRB190718A). Additionally, we can see that after iteration 250, the scatter pattern converges so fast. It is because (after this iteration) the TSNE instance in _scikit Learn_ updates the Kullback–Leibler divergence and `early_exaggeration` parameter.

To do more complex analysis, we can highlight custom GRBs, see redshift dependence in marker size (however, there isn't much redshift info in Swift data), and configure the TSNE running instance. For example, the tSNE convergence setting $215$ in _perplexity_, 'auto' in _learning_rate_, and 'cosine' as metric follows:


```python
file_name = os.path.join('README_files', 'convergence_animation_2.gif')
object1.convergence_animation(features, filename=file_name, perplexity=215, learning_rate='auto', metric='cosine', duration_s=durations)
```

![](README_files/convergence_animation_2.gif)

## tSNE Hyperparameter review

As pioneered by [Wattenberg et al. 2016](https://distill.pub/2016/misread-tsne/), tSNE results cannot be understood only by seeing one scatter plot in 2D. As they said: "_Getting the most from t-SNE may mean analyzing multiple plots with different perplexities._" For this job, you can use the `tsne_animation` instance to iterate over any hyperparameter in sklearn or openTSNE, for example, setting default values in sklearn tSNE and iterating over **perplexity** $\in$ $[5, 500]$:


```python
pp = np.arange(5, 400, 20)
file_name = os.path.join('README_files', 'perplexity_animation.gif')
object1.tsne_animation(features, iterable='perplexity', perplexity=pp, library='sklearn', duration_s=durations, filename=file_name)
```

![](README_files/perplexity_animation.gif)

Note that in some perplexities (i. e. 205), there are "pinched" shapes in the middle plot region. Following [Wattenberg et al. 2016](https://distill.pub/2016/misread-tsne/) analysis: _"chances are the process was stopped too early"_ or this may be because the t-SNE algorithm gets stuck in a bad local minimum.

In general, lower perplexities focus on the substructure of data, and higher perplexities plots are less sensitive to small structures. By contrast, the plot structure does not change globally after perplexity = 245 (except for pinched runs), so we can use this value as default in the following hyperparameters.

The reason why high perplexity values converge better is that noisier datasets (as Swift) will require larger perplexity values to encompass enough local neighbors to see beyond the background noise (see [optimizing tSNE sklearn section](https://scikit-learn.org/stable/modules/manifold.html#t-distributed-stochastic-neighbor-embedding-t-sne)).

Now, we can see what happens if **learning_rate** changes within $10$ and $1000$ (values recommended in [sklearn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html#sklearn.manifold.TSNE)):


```python
lr = np.arange(10, 1000, 75)
object1.tsne_animation(features, duration_s=durations, perplexity=245, filename='learning_rate_animation.gif', iterable='learning_rate', learning_rate=lr)
```

![](README_files/learning_rate_animation.gif)

For _learning_rate_ lower than $200$, the previous global structure preserves, but for some higher values, the tSNE algorithm gets stuck again in a bad local minimum. The conclusion here is that in Swift Data, the learning_rate does not play a relevant role in tSNE convergence.

# Relevant subsets

In this study, we are interested in seeing patterns in tSNE embeddings. Then we searched different relevant subsets of GRBs motivated by the tSNE convergence variation explained in the previous section. In particular, we try to separate two groups (usually named short and long) by their underlying physical process. The following subsections review the main findings made in this task:

## Removing suspicious GRBs
In the [lists of GRBs with special comments](https://swift.gsfc.nasa.gov/results/batgrbcat/summary_cflux/summary_GRBlist), there is some info about failed or partially failed GRB measuring. These GRBs can distract the tSNE algorithm, and fill the spaces between defined groups, broking their general structure.

The GRBs removed are part of the lists:
1. `GRBlist_not_enough_evt_data.txt`:  The event data are only available for part of the burst duration.
2. `GRBlist_tentative_detection_with_note.txt` and `GRBlist_tentative_detection.txt`: GRBs with tentative detection.
3. `Obvious_data_gap.txt`: Obvious data gap within the burst duration.

You can download these tables using the `summary_tables_download` instance:


```python
tables = ('GRBlist_not_enough_evt_data.txt', 'GRBlist_tentative_detection_with_note.txt', 'GRBlist_tentative_detection.txt', 'Obvious_data_gap.txt')
[object1.summary_tables_download(name=name, other=True) for name in tables]
```

Read the tables and index the GRB names:


```python
tables_path = object1.table_path
excluded_names = np.array([])
for table in tables:
    names_i = np.genfromtxt(os.path.join(tables_path, table), usecols=(0, 1), dtype=str)[:, 0]
    excluded_names = np.append(excluded_names, names_i)
excluded_names = np.unique(excluded_names)
print(f"There are {len(excluded_names)} GRBs to be excluded")
```

    There are 116 GRBs to be excluded
    

Remove elements from the original GRB names and features array:


```python
non_match = np.where(np.isin(GRB_names, excluded_names, invert=True))[0]
GRB_names = GRB_names[non_match]
features = features[non_match]
durations = durations[non_match]
print(f"Now there are {len(GRB_names)} GRBs to perform tSNE")
```

    Now there are 1210 GRBs to perform tSNE
    

With these GRB, now the tSNE embedding follows:


```python
tsne_reduced = object1.perform_tsne(features, perplexity=4, init='random', verbose=100)
object1.tsne_scatter_plot(tsne_reduced, duration_s=durations)
```

    [t-SNE] Computing 13 nearest neighbors...
    [t-SNE] Indexed 1210 samples in 0.168s...
    [t-SNE] Computed neighbors for 1210 samples in 4.076s...
    [t-SNE] Computed conditional probabilities for sample 1000 / 1210
    [t-SNE] Computed conditional probabilities for sample 1210 / 1210
    [t-SNE] Mean sigma: 16.434406
    [t-SNE] Computed conditional probabilities in 0.029s
    [t-SNE] Iteration 50: error = 88.6639252, gradient norm = 0.4014133 (50 iterations in 3.106s)
    [t-SNE] Iteration 100: error = 84.0424347, gradient norm = 0.3644353 (50 iterations in 2.738s)
    [t-SNE] Iteration 150: error = 81.1205597, gradient norm = 0.3840486 (50 iterations in 2.450s)
    [t-SNE] Iteration 200: error = 81.4090729, gradient norm = 0.3765063 (50 iterations in 2.619s)
    [t-SNE] Iteration 250: error = 80.7687988, gradient norm = 0.3683825 (50 iterations in 2.704s)
    [t-SNE] KL divergence after 250 iterations with early exaggeration: 80.768799
    [t-SNE] Iteration 300: error = 2.0273073, gradient norm = 0.0027116 (50 iterations in 2.270s)
    [t-SNE] Iteration 350: error = 1.7351900, gradient norm = 0.0010265 (50 iterations in 2.152s)
    [t-SNE] Iteration 400: error = 1.6369183, gradient norm = 0.0004986 (50 iterations in 1.875s)
    [t-SNE] Iteration 450: error = 1.5854030, gradient norm = 0.0004545 (50 iterations in 2.056s)
    [t-SNE] Iteration 500: error = 1.5552773, gradient norm = 0.0002690 (50 iterations in 1.905s)
    [t-SNE] Iteration 550: error = 1.5359926, gradient norm = 0.0002256 (50 iterations in 2.058s)
    [t-SNE] Iteration 600: error = 1.5224233, gradient norm = 0.0001924 (50 iterations in 1.869s)
    [t-SNE] Iteration 650: error = 1.5121573, gradient norm = 0.0001885 (50 iterations in 1.903s)
    [t-SNE] Iteration 700: error = 1.5032912, gradient norm = 0.0001748 (50 iterations in 1.993s)
    [t-SNE] Iteration 750: error = 1.4960387, gradient norm = 0.0001557 (50 iterations in 1.867s)
    [t-SNE] Iteration 800: error = 1.4904945, gradient norm = 0.0001357 (50 iterations in 2.095s)
    [t-SNE] Iteration 850: error = 1.4852208, gradient norm = 0.0001885 (50 iterations in 1.823s)
    [t-SNE] Iteration 900: error = 1.4810259, gradient norm = 0.0001429 (50 iterations in 1.771s)
    [t-SNE] Iteration 950: error = 1.4780662, gradient norm = 0.0001141 (50 iterations in 1.912s)
    [t-SNE] Iteration 1000: error = 1.4752235, gradient norm = 0.0001076 (50 iterations in 1.966s)
    [t-SNE] KL divergence after 1000 iterations: 1.475224
    




    <AxesSubplot:>




    
![png](README_files/README_62_2.png)
    



```python
special_cases = ("GRB060505", "GRB060614", "GRB111209A", "GRB160821B", "GRB130603B")  # Special GRBs, section 4 Jespersen et al. (2020)
special_cases_other = ('GRB050724', 'GRB050911', 'GRB051227', 'GRB060614', 'GRB061006', 'GRB061210', 'GRB070714B',
                       'GRB071227', 'GRB080123', 'GRB080503', 'GRB090531B', 'GRB090715A', 'GRB090916', 'GRB111121A')
tsne_data = object1.perform_tsne(features, library='sklearn', perplexity=110, learning_rate='auto', init='random', n_iter_without_progress=50, verbose=100)
figure_tsne_1 = object1.tsne_scatter_plot(tsne_data, duration_s=durations, names=GRB_names, special_cases=special_cases)
```


```python
tsne_data = object1.perform_tsne(features, library='openTSNE', perplexity=50, initialization="random")
tsne_figure_2 = object1.tsne_scatter_plot(tsne_data, duration_s=durations, names=GRB_names, special_cases=special_cases)
```


```python
special_cases = ("GRB060505", "GRB060614", "GRB111209A", "GRB160821B", "GRB130603B")  # Special GRBs, section 4 Jespersen et al. (2020)
animation = object1.tsne_animation(features, filename='animation.gif', perplexity=np.arange(1, 10, 2), duration_s=durations, names=GRB_names, special_cases=special_cases, verbose=False, library='sklearn')
animation.size = [1920, 1080]
animation.write_gif('animation.gif', fps=2, program='imageio')
```

## Varying Time filter Results
# t_start + 1s


```python
durations_T100_array = object1.durations_checker(GRB_names, t=100)  # Check for name, t_start, and t_end
start_T_100, *other = durations_T100_array[:, :, 1].astype(float), durations_T100_array[:, :, 2].astype(float)
t_start_plus_1 = start_T_100 + 1  # Define the time added
limited_data, GRB_names, errors = object1.so_much_lc_limiters(GRB_names, limits=np.concatenate((start_T_100, t_start_plus_1), axis=1))
normalized_data = object1.so_much_normalize(limited_data)  # Normalizing all light curves
zero_padded_data = object1.so_much_zero_pad(normalized_data)  # Zero-pad data
pre_processing_data = object1.so_much_fourier(zero_padded_data)  # DFT to data
object1.save_data(f"DFT_plus1s_Preprocessed_data_{object1.res}ms", names=GRB_names, data=pre_processing_data)  # Save data
print(f"There are {len(errors)} GRBs that cannot be limited:")  # Print how many errors there are
y = PrettyTable()  # Create printable table
column_names_2 = ('Name', 't_start', 't_end', 'Error Type')
[y.add_column(column_names_2[i], errors[:, i]) for i in range(len(errors[0]))]  # Add rows to each column
print(y)  # Print Errors Table
```


```python
durations_data_array = object1.durations_checker(GRB_names, t=90)  # Check for name, t_start, and t_end
start_times, end_times = durations_data_array[:, :, 1].astype(float), durations_data_array[:, :, 2].astype(float)
durations = np.reshape(end_times - start_times, len(durations_data_array))  # T_90 is equal to t_end - t_start
data_loaded = np.load(os.path.join(object1.results_path, f"DFT_plus1s_Preprocessed_data_{object1.res}ms.npz"))
GRB_names, features = data_loaded['GRB_Names'], data_loaded['Data']
print(f"There are {len(GRB_names)} GRBs loaded: {GRB_names}")
special_cases = ("GRB060505", "GRB060614", "GRB111209A", "GRB160821B", "GRB130603B")  # Special GRBs, section 4 Jespersen et al. (2020)
tsne_data_plus1 = object1.perform_tsne(features, library='sklearn', perplexity=500, learning_rate='auto', init='random', n_iter_without_progress=50)
figure_tsne_plus1 = object1.tsne_scatter_plot(tsne_data_plus1, duration_s=durations, names=GRB_names, special_cases=special_cases)
```

## t_start + 1.5s


```python
durations_T100_array = object1.durations_checker(GRB_names, t=100)  # Check for name, t_start, and t_end
start_T_100, *others = durations_T100_array[:, :, 1].astype(float), durations_T100_array[:, :, 2].astype(float)
t_start_plus_1 = start_T_100 + 1.5  # Define the time added
limited_data, GRB_names, errors = object1.so_much_lc_limiters(GRB_names, limits=np.concatenate((start_T_100, t_start_plus_1), axis=1))
normalized_data = object1.so_much_normalize(limited_data)  # Normalizing all light curves
zero_padded_data = object1.so_much_zero_pad(normalized_data)  # Zero-pad data
pre_processing_data = object1.so_much_fourier(zero_padded_data)  # DFT to data
object1.save_data(f"DFT_plus1_5s_Preprocessed_data_{object1.res}ms", names=GRB_names, data=pre_processing_data)  # Save data
print(f"There are {len(errors)} GRBs that cannot be limited:")  # Print how many errors there are
y = PrettyTable()  # Create printable table
column_names_2 = ('Name', 't_start', 't_end', 'Error Type')
[y.add_column(column_names_2[i], errors[:, i]) for i in range(len(errors[0]))]  # Add rows to each column
print(y)  # Print Errors Table
```


```python
data_loaded = np.load(os.path.join(object1.results_path, f"DFT_plus1_5s_Preprocessed_data_{object1.res}ms.npz"))
GRB_names, features = data_loaded['GRB_Names'], data_loaded['Data']
durations_data_array = object1.durations_checker(GRB_names, t=90)  # Check for name, t_start, and t_end
start_times, end_times = durations_data_array[:, :, 1].astype(float), durations_data_array[:, :, 2].astype(float)
durations = np.reshape(end_times - start_times, len(durations_data_array))  # T_90 is equal to t_end - t_start
print(f"There are {len(GRB_names)} GRBs loaded: {GRB_names}")
special_cases = ("GRB060505", "GRB060614", "GRB111209A", "GRB160821B", "GRB130603B")  # Special GRBs, section 4 Jespersen et al. (2020)
tsne_data_plus1_5 = object1.perform_tsne(features, library='sklearn', perplexity=30, learning_rate='auto', init='random', n_iter_without_progress=50)
figure_tsne_plus1_5 = object1.tsne_scatter_plot(tsne_data_plus1_5, duration_s=durations, names=GRB_names, special_cases=special_cases)
```

# t_start + 2s


```python
durations_T100_array = object1.durations_checker(GRB_names, t=100)  # Check for name, t_start, and t_end
start_T_100, *otherss = durations_T100_array[:, :, 1].astype(float), durations_T100_array[:, :, 2].astype(float)
t_start_plus_1 = start_T_100 + 2  # Define the time added
limited_data, GRB_names, errors = object1.so_much_lc_limiters(GRB_names, limits=np.concatenate((start_T_100, t_start_plus_1), axis=1))
normalized_data = object1.so_much_normalize(limited_data)  # Normalizing all light curves
zero_padded_data = object1.so_much_zero_pad(normalized_data)  # Zero-pad data
pre_processing_data = object1.so_much_fourier(zero_padded_data)  # DFT to data
object1.save_data(f"DFT_plus2s_Preprocessed_data_{object1.res}ms", names=GRB_names, data=pre_processing_data)  # Save data
print(f"There are {len(errors)} GRBs that cannot be limited:")  # Print how many errors there are
y = PrettyTable()  # Create printable table
column_names_2 = ('Name', 't_start', 't_end', 'Error Type')
[y.add_column(column_names_2[i], errors[:, i]) for i in range(len(errors[0]))]  # Add rows to each column
print(y)  # Print Errors Table
```


```python
durations_data_array = object1.durations_checker(GRB_names, t=90)  # Check for name, t_start, and t_end
start_times, end_times = durations_data_array[:, :, 1].astype(float), durations_data_array[:, :, 2].astype(float)
durations = np.reshape(end_times - start_times, len(durations_data_array))  # T_90 is equal to t_end - t_start
data_loaded = np.load(os.path.join(object1.results_path, f"DFT_plus2s_Preprocessed_data_{object1.res}ms.npz"))
GRB_names, features = data_loaded['GRB_Names'], data_loaded['Data']
print(f"There are {len(GRB_names)} GRBs loaded: {GRB_names}")
special_cases = ("GRB060505", "GRB060614", "GRB111209A", "GRB160821B", "GRB130603B")  # Special GRBs, section 4 Jespersen et al. (2020)
tsne_data_plus1_5 = object1.perform_tsne(features, library='sklearn', perplexity=500, learning_rate='auto', init='random', n_iter_without_progress=50)
figure_tsne_plus1_5 = object1.tsne_scatter_plot(tsne_data_plus1_5, duration_s=durations, names=GRB_names, special_cases=special_cases)
```


```python
output = os.system('jupyter nbconvert  README.ipynb --to markdown --output README.md')
output2 = os.system('jupyter nbconvert  README.ipynb --to html --output README.html')
print(f"Readable file created") if output == 0 else None
```

    Readable file created
    


```python

```
