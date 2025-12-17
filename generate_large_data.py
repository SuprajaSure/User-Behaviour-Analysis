import pandas as pd
import numpy as np

n = 10000
data = pd.DataFrame({
    'user_id': np.arange(1, n+1),
    'session_duration': np.random.normal(30, 10, n),
    'pages_visited': np.random.normal(10, 3, n),
    'click_rate': np.random.normal(0.5, 0.1, n),
    'time_of_day': np.random.randint(0, 24, n),
    'location': np.random.choice(['India', 'US', 'UK', 'Germany', 'Japan'], n),
    'device_type': np.random.choice(['Mobile', 'Desktop', 'Tablet'], n)
})

for i in np.random.choice(data.index, 300, replace=False):
    data.loc[i, 'session_duration'] *= np.random.randint(4, 8)
    data.loc[i, 'click_rate'] *= np.random.uniform(2, 5)

data.to_csv("large_user_data.csv", index=False)
print("✅ large_user_data.csv created successfully!")
