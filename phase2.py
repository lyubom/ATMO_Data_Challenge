import numpy as np
import random
import matplotlib.pyplot as plt
from utils import load_data, get_data_day, get_common_sites

def convert_pol(pol):
    if pol == "PM10":
        return 4
    elif pol == "PM25":
        return 3
    elif pol == "NO2":
        return 2
    elif pol == "O3":
        return 1



def analyze_var():

    data = np.load('./tmp/final_data_2.npz')
    X, Y = data['a'], data['b']

    pol_data = dict()
    pol_plot_data = dict()

    for pol in ["PM10","PM25","O3","NO2"]:
        p = convert_pol(pol)

        idx = np.random.randint(300088, size=50)
        X_ = X[idx]

        pol_data[pol] = X_[X_[:,0] == p][:,3:-1]

        pol_t_minus_1 = pol_data[pol][:, 0:-2]
        pol_t = pol_data[pol][:, 2:]



        pol_t_minus_1 = pol_t_minus_1.flatten()
        pol_t = pol_t.flatten()

        plt.subplot(2, 2, p)
        plt.plot(pol_t_minus_1, pol_t, 'r.')
        plt.title(pol)

    plt.show()

def analyze_covar(hours):


    all_data = load_data(dirname="../Data/training", year=2016)

    common_sites = get_common_sites(all_data)
    site = common_sites.pop()

    data_pol = dict()

    days = np.unique(np.random.randint(low=2, high=360, size=50))
    print(days)

    for day in days:

        data_day = get_data_day(day, all_data, max_days=1, year=2016)
        # print(site)

        for pol in ["PM10","PM25","O3","NO2"]:
            # print(pol)

            concentrations_pol_site = data_day[3][pol][data_day[3][pol].idPolair==site]
            previous_data = np.array(concentrations_pol_site.Valeur)

            if np.isnan(previous_data).any():
                # print('True')
                if site != '33374':
                    chimeres_site = data_day[0][pol].loc[data_day[0][pol].idPolair == float(site)]
                else:
                    chimeres_site = data_day[0][pol].loc[data_day[0][pol].idPolair == 15114.]

                inds = np.where(np.isnan(previous_data))

                previous_data[inds] = chimeres_site['val'].iloc[inds]

            if day == days[0]:
                data_pol[pol] = np.array([previous_data[0:hours]])
            else:

                data_pol[pol] = np.append(data_pol[pol], [previous_data[0:hours]], axis=0)


    for y in range(hours):
        for x in range(hours):
            ax = plt.subplot(hours, hours, y*hours + x + 1)
            for pol in ["PM10","PM25","O3","NO2"]:

                yaxis = data_pol[pol][:, y]
                xaxis = data_pol[pol][:, x]


                format = 'C' + str(convert_pol(pol)) + '.'
                ax.plot(xaxis, yaxis, format, label=pol)
    ax.legend()
    plt.show()

analyze_var()
analyze_covar(4)
# analyze_covar(24)
