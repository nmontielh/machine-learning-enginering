import pandas as pd
import matplotlib.pyplot as plt

#get_ipython().run_line_magic('matplotlib', 'inline')

baby_name = ['Hugo','Paco','Luis','Donald']
number_birts = [96, 155, 66, 272]

data_zip = zip(baby_name, number_birts)
data_set = list(data_zip)
data_frame = pd.DataFrame(data = data_set, columns=['Name','Number'])

# No funciona esta en python solo
data_frame['Number'].plot()

plt.plot(data_frame['Number']);
plt.legend();
plt.show();