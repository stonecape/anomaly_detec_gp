
# coding: utf-8

# In[43]:

# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'
import matplotlib
matplotlib.use('agg')

from random import *
from matplotlib import pyplot as plt
import numpy as np
import GPy
from datetime import datetime
start_time = datetime.now()
# print("start_time ",start_time)


# the class name we want to model
model_class_name = 3

# randomly sample demo_light_num rows to model our function
# demo_light_num = 30;

demo_light_percentage = 0.3;

# sample sample_time_index_num time-indexes/columns from the data (1024 columns totally)
sample_time_index_num = 200;

lightcurve_train_set_path = "./StarLightCurves/StarLightCurves_TRAIN"
lightcurve_test_set_path = "./StarLightCurves/StarLightCurves_TEST"


# Due to the memory error, we need to reduce the number of time indexes in this practice. 

# In[44]:

# sample_time_index_num time indexes from 1024 indexes
times_indexes = np.asarray(sorted(sample(list(range(0, 1024)), sample_time_index_num)))
# times_indexes = np.arange(0,1024, 1024/sample_time_index_num, dtype=int);
print("We choose the following ",sample_time_index_num, "time indexes from 1024 ones\n", times_indexes)


# In[45]:

# class name {1,2,3}
class_names = np.loadtxt(lightcurve_train_set_path, delimiter=',', usecols=[0])

# light curve data
light_curve = np.loadtxt(lightcurve_train_set_path, delimiter=',', usecols=range(1, 1025))

# normalize magnitude
light_curve -= light_curve.mean(1)[:,np.newaxis]
light_curve /= light_curve.std(1)[:,np.newaxis]


# In[46]:

indices = [i for i,cn in enumerate(class_names) if cn == model_class_name]
print("model_class_name=",model_class_name," totally has ", len(indices), " rows");

demo_light_num = int(demo_light_percentage * len(indices))
indices = sorted(sample(indices, demo_light_num))
# indices = indices[0:demo_light_num]
print("We select the following ",len(indices)," rows\n",indices)


# In[47]:

#construct a hierarchical GPy kernel. 
kern_class = GPy.kern.Matern32(input_dim=1, variance=1.5, lengthscale=2.5, active_dims=[0], name='class')
kern_replicate = GPy.kern.Matern32(input_dim=1, variance=.1, lengthscale=2.5, active_dims=[0], name='replicate')
k_hierarchy = GPy.kern.Hierarchical(kernels=[kern_class, kern_replicate])
print(k_hierarchy)


# In[48]:

print(np.ones(sample_time_index_num).shape)
print(times_indexes.reshape(-1,1).shape)


# In[49]:

time_index_stack = np.tile(times_indexes.reshape(-1,1),(demo_light_num, 1)) ;
print(time_index_stack.shape)


# In[50]:

replicate_stack = []
for r in range(1, demo_light_num + 1):
    replicate_stack.append(np.ones(sample_time_index_num) * r);
    
replicate_stack = np.asarray(replicate_stack).reshape(-1,1);
print(replicate_stack.shape)


# In[51]:

filtered_light_curve = light_curve[indices]
filtered_light_curve = filtered_light_curve[:,times_indexes]
print(filtered_light_curve.shape)


# X:
# time_index, replicate_number
# 
# Y:
# light_curve_magnitude

# In[52]:

X = np.hstack((time_index_stack, replicate_stack))
X = np.vstack(X)
Y = np.vstack(filtered_light_curve.reshape(-1,1))

print(X.shape)
print(Y.shape)


# In[67]:

m = GPy.models.GPRegression(X=X, Y=Y, kernel=k_hierarchy)
# m.optimize_restarts(num_restarts=5);
m.optimize('bfgs', messages=1)
print(k_hierarchy)


# Here, we plot the top 10 rows data.

# In[54]:

plt.figure(figsize=(20,1.85))
#to plot the mean function g_n(t), predict using only the 'kern_class' kernel

Xplot = np.linspace(1,1025, 100)[:,None]
mu, var = m.predict(Xplot, kern=kern_class)
ax = plt.subplot(1,11,1)
GPy.plotting.matplot_dep.base_plots.gpplot(Xplot, mu, mu - 2*np.sqrt(var), mu + 2*np.sqrt(var), ax=ax, edgecol='r', fillcol='r')
plt.ylabel('normalized magnitude')
plt.xlabel('underlying\nfunction')

#plot each of the functions f_{nr}(t)
for r in range(1,11):
    ax = plt.subplot(1,11,r+1)
    m.plot(fixed_inputs=[(1,r)],ax=ax, which_data_rows=X[:,1]==r, legend=False)
    plt.xlabel('star %i'%r)
    plt.plot(Xplot, mu, 'r--', linewidth=1)

    
GPy.plotting.matplot_dep.base_plots.align_subplots(1,11, xlim=(1,1000), ylim=(-3,3))

#fig_name = 'result' + datetime.now().strftime("%Y-%m-%d %H-%M") + '.png'
fig_name = "model_class_name_" + str(model_class_name);
plt.savefig(fig_name);


# In[55]:

# x_test = np.arange(1,1025)[:,None]
# y_test = light_curve_test[0,:].reshape(-1,1)
# print(x_test.shape, y_test.shape)


# In[56]:

# mu_star, var_star = m.predict_noiseless(x_test, kern=kern_class) # read the source code for log_predictive
# m.likelihood.log_predictive_density(y_test, mu_star, var_star)


# In[57]:

class_names_test = np.loadtxt(lightcurve_test_set_path, delimiter=',', usecols=[0])

light_curve_test = np.loadtxt(lightcurve_test_set_path, delimiter=',', usecols=range(1, 1025))
light_curve_test -= light_curve_test.mean(1)[:,np.newaxis]
light_curve_test /= light_curve_test.std(1)[:,np.newaxis]

test_indices = [i for i,cn in enumerate(class_names_test) if cn == model_class_name]
# print("model_class_name=",model_class_name," totally has ", len(test_indices), " rows in test dataset");


# In[58]:

log_pre_density_result = np.ones(len(test_indices)) * 9999
log_pre_density_result = log_pre_density_result.reshape(-1,1)
# print(log_pre_density_result.shape)


# sum /avg the log_predictive_density?

# In[59]:

x_test = np.arange(1,1025)[:,None]
mu_star, var_star = m.predict_noiseless(x_test, kern=kern_class) # read the source code for log_predictive
for index in range(len(test_indices)):
    y_test = light_curve_test[test_indices[index],:].reshape(-1,1)
    log_pre_density_result[index] = np.average(m.likelihood.log_predictive_density(y_test, mu_star, var_star))


# In[60]:

test_indices = np.asarray(test_indices)
test_indices = test_indices.reshape(-1,1)
#print(log_pre_density_result.shape, test_indices.shape)
# print(log_pre_density_result.shape, test_indices.shape)


# In[61]:

combine_result = np.concatenate((test_indices, log_pre_density_result), axis=1)
sorted_result = np.sort(combine_result.view('i8,i8'), order=['f1'], axis=0).view(np.float)
print(sorted_result)


# In[66]:

result_file_name = "sorted_result_" + str(model_class_name) + ".csv";
np.savetxt(result_file_name, sorted_result, delimiter=",",fmt='%d,%1.9f')


# In[ ]:



