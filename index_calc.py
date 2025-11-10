import numpy as np
from collections import Counter

# def calc_distribution(time_series):
#     divisors=time_series[1:]-time_series[0:-1]
#     #print(divisors)
#     correct_sample=[]
#     check=0
#     mean_divisors=np.mean(divisors)
#     print("mean")
#     samples = divisors[divisors<=mean_divisors]
#     double_divided={}

#     for i in range(len(samples)):
#         divided = np.rint(time_series / samples[i]).astype(int)
#         double_divided[i]=divided

#     arrays_as_tuples = [tuple(arr) for arr in double_divided.values()]
#     freq_counter = Counter(arrays_as_tuples)
#     most_common_array, max_freq = freq_counter.most_common(1)[0]
#     #print(f"Most frequent array: {list(most_common_array)} (Frequency: {max_freq})")
#     keys_with_most_common = [k for k, v in double_divided.items() if tuple(v) == most_common_array]
#     print(samples[keys_with_most_common])
#     mean=np.min(samples[keys_with_most_common])
#     std=np.std(time_series)
#     print(max_freq)
#     if(max_freq>=3):
#         return mean,std,np.rint(time_series / mean).astype(int)k
#     else:
#         print("Not enough freq. Require manual analysis")

def calc_distribution(time_series):
    divisors=time_series[1:]-time_series[0:-1]
    mean_divisors=np.mean(divisors)
    print("mean_divisors",mean_divisors)
    samples = divisors[divisors<=(mean_divisors*1.5)]
    indices_of_break = np.sort(np.where(divisors >= (mean_divisors * 1.5))[0])
    # print(f"Length of time seriees = {len(time_series)}\n")
    # print(indices_of_break)
    mean_samples=np.mean(samples)
    std=np.std(divisors)
    index_to_return = np.arange(0,len(time_series))

    for index in indices_of_break:
        val=time_series[1+index]-time_series[index]
        increment=int(np.rint(val/mean_samples))
        index_to_return[index+1:]+=(increment-1)
        #print(val)

    return mean_samples,std,index_to_return
  

    



    #     print("len\n")
    #     print(divided)
    #     if np.array_equal(divided, temp):
    #         check+=1
    #         print(i)
    #         correct_sample.append(i)
    #         temp=divided
    # mean=np.mean(correct_sample)
    # std=np.std(time_series)
    # if(check>=3):
    #     return mean,std,np.rint(time_series / mean).astype(int)




