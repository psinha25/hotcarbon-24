import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset

# Source: https://lambdalabs.com/service/gpu-cloud#pricing
RTX_COST = 0.50 # price/hr for an RTX 6000 24 GB
A100_COST = 1.79 # price/hr for an A100 80GB
A6000_COST = 0.80 # price/hr for an A6000

RTX_EMBODIED_CARBON = 1/0.9 * (583 * 2.75 + 430 + 500) * 6.09 + 66.60 * 24 + 2 * 150
A100_EMBODIED_CARBON = 1/0.9 * (583 * 1.52 + 350 + 500) * 8.26 + 66.60 * 80 + 2 * 150 
A6000_EMBODIED_CARBON  = 1/0.9 * (583 * 2.15 + 350 + 500) * 6.28 + 66.60 * 48 + 2 * 150

GPU_LIFETIME = 4 * 352 * 24 * 60 * 60 # years * days/year * hours/day * minutes/hour * seconds/minute

COST_CARBON_DICT = {'4090': [RTX_COST, RTX_EMBODIED_CARBON], 'a100': [A100_COST, A100_EMBODIED_CARBON], 'a6000': [A6000_COST, A6000_EMBODIED_CARBON]}

def aggregate_results():

    configuration = glob("*/*/*")
    devices = []
    mixes = []
    energy = []
    latency_stats = []
    throughput = []
    for c in configuration:
        
        pwr = pd.read_csv(c+"/pwr.csv")
        if c.startswith("a100"):
            pwr = pwr[0::4]
        elif c.startswith("a6000"):
            pwr = pwr[0::8]
        elif c.startswith("4090"):
            pass
        else:
            assert(1==0)

        pwr = pwr[-301:]
        t = pd.to_datetime(pwr['timestamp'], format='%Y/%m/%d %H:%M:%S.%f')
        δt = (np.diff(t)/1e9).astype("float")
        P = np.array(pwr[' power.draw [W]'][:-1].str.replace(' W', '')).astype(float)
        energy.append(np.sum(δt*P))

        tput = pd.read_csv(c+"/tput.csv").to_numpy()[0][2:]
        p0 = pd.read_csv(c+"/total_p0.csv").to_numpy()[0][2:]
        p50 = pd.read_csv(c+"/total_p50.csv").to_numpy()[0][2:]
        p90 = pd.read_csv(c+"/total_p90.csv").to_numpy()[0][2:]
        p100 = pd.read_csv(c+"/total_p100.csv").to_numpy()[0][2:]

        latency = np.vstack([p0,p50,p90,p100])

        while len(tput) <3:
            tput = np.hstack([tput,np.nan])
            latency = np.hstack([latency,[[np.nan],[np.nan],[np.nan],[np.nan]]])
        
        devices.append(c.split('/')[0])
        mixes.append(c.split('/')[1])
        latency_stats.append(latency)
        throughput.append(tput)
    
    dataset = Dataset.from_dict({
        'device': devices,
        'mix': mixes,
        'energy': energy,
        'latency_stats': latency_stats,
        'throughput': throughput,
        'configuration': configuration,
    })
    return dataset
    
def construct_df(dataset: Dataset):

    df_rows = [] # mix, device, sharing_style, num_inferences, latency, energy, embodied carbon, cost, GPU hours

    for gpu in ['4090', 'a100']:
        device_rows = dataset.filter(lambda x: x['device'] == '4090')
        for row in device_rows:
            jobs = row['mix'].split('-')
            if len(jobs) > 1:
                
                # Calculate number completed while sharing in 5 minutes
                num_reqs = 0
                num_reqs_per_job = []
                for i in range(len(jobs)):
                    num_reqs += 60*5*row['throughput'][i]
                    num_reqs_per_job.append(60*5*row['throughput'][i])
                single_row = ["\n".join(jobs), 
                              gpu, 
                              'Single GPU', 
                              num_reqs, 
                              300, 
                              row['energy'], 
                              (COST_CARBON_DICT[gpu][1] * 60 * 5) / GPU_LIFETIME,  
                              (COST_CARBON_DICT[gpu][0] * 60 * 5) / 3600,
                              5 / 60]

                
                
                # Construct multi-GPU row for mix
                latency = -1
                energy = 0
                embodied_carbons = []
                costs = []
                latencies = []
                for i, job in enumerate(jobs):
                    single_model_row = device_rows.filter(lambda y: y['mix'] == job)[0]
                    curr_latency = single_model_row['latency_stats'][3][0] / 1000
                    energy += single_model_row['energy'] / (single_model_row['throughput'][0] * 60 * 5) * num_reqs_per_job[i]
                    embodied_carbons.append((COST_CARBON_DICT[gpu][1] * curr_latency * num_reqs_per_job[i]) / GPU_LIFETIME)
                    costs.append((COST_CARBON_DICT[gpu][0] * curr_latency * num_reqs_per_job[i]) / 3600)
                    latencies.append(curr_latency * num_reqs_per_job[i])

                    if curr_latency * num_reqs_per_job[i] > latency:
                        latency = curr_latency * num_reqs_per_job[i]
                multi_row = ["\n".join(jobs), gpu, 'GPU/model', num_reqs, latency, energy, sum(embodied_carbons), sum(costs), sum(latencies) / 3600]
                
                df_rows.append(single_row)
                df_rows.append(multi_row)
        
    df = pd.DataFrame(df_rows, columns=['mix', 'device', 'sharing_style', 'num_inferences', 'latency', 'energy', 'embodied_carbon', 'cost', 'gpu_hours'])
    return df

def plot_grid(df: pd.DataFrame):
    print(df)
    for device in ['4090', 'a100']:
        df_device = df[df['device'] == device]
        sns.set_style("whitegrid", {'grid.linestyle': ':'})
        fig, axs = plt.subplots(2, 2, figsize=(13, 6), constrained_layout = True)

        # Latency plot
        p1 = sns.barplot(data=df_device, x='mix', y='latency', hue='sharing_style', ax=axs[0, 0])
        p1.set_xlabel('')
        p1.set_ylabel('Latency (s)')

        # Cost plot
        p2 = sns.barplot(data=df_device, x='mix', y='gpu_hours', hue='sharing_style', ax=axs[0, 1])
        p2.set_xlabel('')
        p2.set_ylabel('GPU Hours')

        # Operational Energy Plot
        p3 = sns.barplot(data=df_device, x='mix', y='energy', hue='sharing_style', ax=axs[1, 0])
        p3.set_xlabel('')
        p3.set_ylabel('Energy (joules)')

        # Embodied Carbon
        p4 = sns.barplot(data=df_device, x='mix', y='embodied_carbon', hue='sharing_style', ax=axs[1, 1])
        p4.set_xlabel('')
        p4.set_ylabel('Embodied Carbon (g CO2)')

        plt.savefig(f'./plots/{device}.pdf', bbox_inches='tight', dpi=400, format='pdf')
        plt.close()
    




if __name__=='__main__':
    dataset = aggregate_results()
    df = construct_df(dataset=dataset)
    plot_grid(df=df)