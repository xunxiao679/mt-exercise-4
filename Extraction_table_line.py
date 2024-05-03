import re
import pandas as pd
import matplotlib.pyplot as plt

def extract_perplexities(log_file):
    perplexities = []
    with open(log_file, 'r') as file:
        for line in file:
            line = line.strip()  # Remove leading/trailing whitespace
            if line:
                pattern = r'ppl:\s+(\d+\.\d+)'
                match = re.search(pattern, line)
                if match:
                    ppl = float(match.group(1))
                    perplexities.append(ppl)
    return perplexities

# Define log file paths
baseline_log_file = 'baseline.log'
pre_norm_log_file = 'log_pre_post/deen_transformer_pre/deen_transformer_pre_log'
post_norm_log_file = 'log_pre_post/deen_transformer_post/deen_transformer_post_log'

# Extract perplexities
baseline_perplexities = extract_perplexities(baseline_log_file)
pre_norm_perplexities = extract_perplexities(pre_norm_log_file)
post_norm_perplexities = extract_perplexities(post_norm_log_file)

print("Baseline perplexities:", baseline_perplexities)
print("Pre-Norm perplexities:", pre_norm_perplexities)
print("Post-Norm perplexities:", post_norm_perplexities)

# Create a DataFrame
data = {
    'Validation ppl': range(500, len(baseline_perplexities)*500 + 1, 500),
    'Baseline': baseline_perplexities,
    'Prenorm': pre_norm_perplexities,
    'Postnorm': post_norm_perplexities
}
df = pd.DataFrame(data)
df.to_csv('validation_ppl.csv', index=False)

# Create a line chart
plt.figure(figsize=(10, 6))
plt.plot(df['Validation ppl'], df['Baseline'], label='Baseline', color='blue')
plt.plot(df['Validation ppl'], df['Prenorm'], label='Pre-Norm', color='green')
plt.plot(df['Validation ppl'], df['Postnorm'], label='Post-Norm', color='red')

plt.title('Validation Perplexities of Transformer Models')
plt.xlabel('Validation ppl')
plt.ylabel('Perplexities')
plt.legend()
plt.grid(True)
plt.show()
